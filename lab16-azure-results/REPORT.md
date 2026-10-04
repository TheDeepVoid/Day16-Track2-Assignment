# LAB 16 — Cloud AI Environment Setup (Azure track) — Báo cáo kết quả

**Ngày chạy:** 2026-10-03 · **Subscription:** Azure for Students
**Resource group:** `ai-lab-rg` (Malaysia West) · **VM:** `ai-cpu-node` — `Standard_B2s_v2`
**Hệ điều hành:** Ubuntu 24.04 LTS (kernel 6.17.0-1022-azure)

---

## 1. Bảng kết quả (Part 4.4)

| Metric | Kết quả |
|---|---|
| Thời gian load data | **1.070 s** (143.8 MB → 134.4 MB/s) |
| Thời gian training | **8.644 s** (400 cây, 2 vCPU) |
| Best iteration | **400** (chọn bằng 5-fold cross-validation) |
| AUC-ROC | **0.940221** (test) — **0.887544 ± 0.012421** (5-fold CV) |
| Accuracy | **0.988413** |
| F1-Score | **0.210526** (tại ngưỡng 0.5) |
| Precision | **0.119241** (tại ngưỡng 0.5) |
| Recall | **0.897959** (tại ngưỡng 0.5) |
| Inference latency (1 row) | **0.7080 ms** |
| Inference throughput (1000 rows) | **93.574 rows/s** (10.687 ms / batch 1000) |

Thời gian cross-validation để chọn `n_estimators`: 40.17 s (tính riêng, không cộng vào "training time").

Tại ngưỡng tối ưu F1 (0.95): F1 = 0.2334, Precision = 0.1341, Recall = 0.8980.

**Output đầy đủ của `benchmark.py`:**

![Output benchmark.py](../screenshots/01-benchmark-output.png)

---

## 2. Ba phát hiện quan trọng khi chạy trên dataset này

Kết quả ban đầu sai rõ rệt (AUC chỉ 0.936, `Best iteration = 1`, F1 = 0.166). Ba nguyên nhân đã được kiểm chứng và xử lý:

**(1) Cột `Time` phải bị loại bỏ.** Đây là timestamp thô, không mang tín hiệu dự đoán. 5-fold CV: **0.897 không có** `Time` so với **0.880 có** `Time`.

**(2) `is_unbalance=True` là bắt buộc.** Dataset chỉ có **0.1727%** gian lận (492/284.807). Không cân bằng class, LightGBM học trên ~400 mẫu dương và overfit rất nhanh. Đây là nguyên nhân trực tiếp khiến `Best iteration = 1`.

**(3) Early stopping trên một validation split nhỏ bị hỏng — lỗi quan trọng nhất.** Validation split chỉ chứa **~79 mẫu gian lận**, nên AUC trên split đó gần như ngẫu nhiên:

| n_estimators | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| AUC (không cân bằng) | 0.9415 | **0.6855** | 0.7771 | 0.7880 | 0.8203 |
| AUC (`is_unbalance`) | 0.9113 | **0.2498** | 0.8974 | 0.8862 | 0.8854 |
| AUC (`scale_pos_weight=100`) | 0.9590 | **0.1133** | 0.9056 | 0.9348 | 0.9183 |

AUC sụp xuống dưới 0.5 ở vòng 2 dưới mọi cách cân bằng, và điều này lặp lại giống hệt nhau trên LightGBM 4.6.0 lẫn 4.7.0 — xác nhận đây là đặc tính của dữ liệu, không phải lỗi thư viện. Do đó validation AUC đạt đỉnh giả ở vòng 1, early stopping dừng ngay, và `predict_proba` chỉ dùng **1 cây**.

Ngoài ra, AUC do chính LightGBM ghi vào lịch sử eval cũng không đáng tin trên dataset này: đường cong bị đóng băng ở đúng một giá trị từ vòng ~450 trở đi. Vì vậy benchmark **tự tính AUC bằng sklearn trên dự đoán thật**, mỗi fold chỉ fit một lần rồi dùng `num_iteration` để lấy dự đoán tại từng mốc.

Kết quả sweep (5-fold, AUC tính bằng sklearn) — đường cong tăng đều rồi đi ngang:

| n_estimators | 50 | 100 | 150 | 200 | 300 | **400** | 500 | 600 |
|---|---|---|---|---|---|---|---|---|
| Mean CV-AUC | 0.8732 | 0.8758 | 0.8795 | 0.8804 | 0.8811 | **0.8875** | 0.8875 | 0.8875 |

Mỗi fold được fit đủ 600 cây, nên ba cột cuối bằng nhau là bão hoà thật chứ không phải bị cắt. Điểm 400 là mốc cuối cùng còn cải thiện, nên được chọn.

---

## 3. Về F1 / Precision thấp

Precision = 0.119 và F1 = 0.211 trong khi AUC = 0.940 là kết quả đúng. Với chỉ 394 mẫu gian lận trong tập huấn luyện và ngưỡng cắt cố định 0.5, mô hình ưu tiên bỏ sót ít hơn là báo động giả ít: Recall = 0.898 nhưng Precision thấp.

`is_unbalance=True` đẩy toàn bộ điểm dự đoán lên cao, nên 0.5 vốn đã là một ngưỡng vận hành kém. Ngưỡng tối ưu F1 tìm được là **0.95** (F1 = 0.2334). Muốn cải thiện đáng kể cần thu thập thêm dữ liệu gian lận hoặc kỹ thuật resampling (SMOTE) — SMOTE không dùng ở đây vì làm sai lệch thật trên tập test.

---

## 4. Tài nguyên & chi phí (Part 5)

### 4.1. Tài nguyên

**CPU** — `Intel Xeon Platinum 8370C @ 2.80GHz`, 2 vCPU (1 core × 2 thread).

- Lúc idle: **0.5 – 7.6%**
- Lúc chạy `benchmark.py`: **95–100%** cả 2 vCPU (tiến trình `python` dùng **200.0% CPU**)
- Azure Monitor (PT5M): **peak 73.6%**, mean 29.8%

**RAM** — 7.8 GiB tổng; khi train chỉ dùng ~1.25 GiB (~16%), còn 6.7–6.9 GiB available. Dataset 284k×29 nằm gọn trong RAM, không cần swap.

**Network** — `ip -s link` trên `eth0`: **RX ~845 MB** (tải dataset 143.8 MB + cài đặt package), **TX ~11 MB**. 0 error, 0 dropped packet. NSG chỉ mở port 22 từ một IP duy nhất, không mở cổng vLLM 8000.

**Disk** — `/dev/root` 29 GB, đã dùng 4.7 GB (17%).

![top khi huấn luyện](../screenshots/02-resource-under-load.png)

![Resource usage](../screenshots/03-resource-usage.png)

![Azure Monitor metrics](../screenshots/04-azure-monitor-metrics.png)

### 4.2. Chi phí

Số liệu thực tế từ Azure Cost Management API (`ActualCost`, `MonthToDate`, granularity `Daily`), truy vấn ngày 2026-10-04, đối chiếu với ảnh Cost Analysis trên Azure Portal.

| Metric / Setting | Value |
|---|---|
| Time Period | Oct 2026 |
| **Actual Cost (USD)** | **0.450525** |
| Forecast | Unavailable (`--`) |
| Budget | None |
| Location | Malaysia West |
| Subscription | Azure for Students |

**Theo Resource Group:**

| Resource Group | PreTaxCost (USD) |
|---|---|
| `ai-lab-rg` | 0.447537 |
| `defaultresourcegroup-eus` | 0.002989 |
| **Tổng** | **0.450525** |

**Theo Resource Group × Service:**

| Resource Group | Service | Cost (USD) |
|---|---|---|
| `ai-lab-rg` | Storage | 0.210267 |
| `ai-lab-rg` | Virtual Network | 0.150435 |
| `ai-lab-rg` | Virtual Machines | 0.086834 |
| `ai-lab-rg` | Bandwidth | 0.000001 |
| `defaultresourcegroup-eus` | Azure Monitor | 0.002989 |
| **Tổng** | | **0.450525** |

**Theo ngày:**

| UsageDate | Resource Group | PreTaxCost (USD) |
|---|---|---|
| 2026-10-02 | `ai-lab-rg` | 0.224831 |
| 2026-10-03 | `ai-lab-rg` | 0.222705 |
| 2026-10-03 | `defaultresourcegroup-eus` | 0.002989 |
| **Tổng** | | **0.450525** |

**Nhận xét:** tổng chi phí thực tế của lab là **0.450525 USD**, trong đó **99.3% (0.447537 USD) phát sinh trong resource group `ai-lab-rg`**; phần còn lại 0.002989 USD là Azure Monitor trong `defaultresourcegroup-eus`, không thuộc lab. Ba dòng chi phí lớn đều nằm trong `ai-lab-rg`: Virtual Machines (compute của `ai-cpu-node`), Virtual Network (VNet + Public IP Standard) và Storage (managed disk của VM). So với ước tính ~$0.05/giờ trong tài liệu giao, tổng 0.450525 USD tương ứng khoảng **9 giờ** VM, phù hợp với thời gian dựng hạ tầng, cài đặt, tải dataset và huấn luyện.

Azure tính hóa đơn theo cơ chế **accrual**, nên dữ liệu 24–72 giờ gần nhất là *provisional* và được tính lại khi usage data hoàn tất. Vì vậy lần truy vấn ngày 2026-10-03 cho hai ngày 02–03-10 cho tổng `ai-lab-rg` là **0.333692 USD**, còn truy vấn lại ngày 2026-10-04 cho cùng hai ngày đó là **0.447536 USD** (ngày 03-10 tăng từ 0.108861 lên 0.222705 USD). Số liệu trình bày ở trên là lần truy vấn đầy đủ ngày 2026-10-04 và khớp chính xác với tổng $0.45 hiển thị trên Portal.

![Azure Cost Analysis — Portal](../screenshots/05-cost-management.png)

---

## 5. Bảo mật

```
Name               Prio  Access  Source              DestPort  Protocol
allow-ssh-from-me  100   Allow   <my-public-ip>/32   22        Tcp
```

NSG chỉ có một rule: SSH port 22, giới hạn đúng `/32` của một IP cá nhân. Không có rule nào mở `0.0.0.0/0`, không mở port 8000. Public IP là **Static** (Standard SKU).

Public IP Standard được khuyến nghị trong tài liệu giao với giá ~$0.005/giờ; trong cấu hình thực tế IP là Static nên không phát sinh phí IP theo giờ, chi phí chỉ đến từ VM và tài nguyên mạng đi kèm.

![Hạ tầng Azure](../screenshots/06-infra-summary.png)

---

## 6. Deliverables (Part 6)

| # | Deliverable | File |
|---|---|---|
| 1 | Screenshot / output terminal đầy đủ | `../screenshots/01-benchmark-output.png` (bản gốc: `benchmark_output.txt`) |
| 2 | File kết quả | `benchmark_result.json` |
| 3 | Resource usage | `../screenshots/02-resource-under-load.png`, `03-resource-usage.png`, `04-azure-monitor-metrics.png` (bản gốc: `resource_usage.txt`, `resource_usage_under_load.txt`, `azure_monitor_metrics.txt`) |
| 4 | Cost Management | `../screenshots/05-cost-management.png` (bản gốc: `cost_portal_analysis.txt`, `cost_terminal_capture.txt`, `cost_data.txt`) |
| 5 | cloud-init đã dùng | `cloud-init-cpu.yaml` |
| 6 | Báo cáo nhận xét | `REPORT.md` (file này) |
| — | Script benchmark | `../benchmark.py` |
| — | Hạ tầng đã dựng | `../screenshots/06-infra-summary.png` (bản gốc: `infra_summary.txt`) |

---

## 7. Sai lệch so với tài liệu giao

| Tài liệu ghi | Thực tế |
|---|---|
| VM `Standard_B2s` | **`Standard_B2s_v2`** (4 GB → 8 GB) |
| Location `eastus` | **`malaysiawest`** (giá và quota khác) |
| `pip3 install ... kaggle` → CLI `kaggle` dùng được | CLI không có trong PATH; `/usr/local/bin/kaggle` là symlink hỏng trỏ tới `/opt/ml-env/bin/kaggle` (venv rỗng). Phải tự tạo venv và dùng `~/ml-env/bin/python` |
| Ước tính giá theo East US | Số liệu thực tế Malaysia West: **0.450525 USD** (§4.2) |
| Early stopping → "Best iteration" | **Không dùng** — xem §2(3). `n_estimators` chọn bằng 5-fold CV |

Cloud-init `pip3 install` trên Ubuntu 24.04 cài package vào `~/.local` (PEP 668), còn CLI `kaggle` cần venv riêng. Để tái lập:

```bash
python3 -m venv ~/ml-env
~/ml-env/bin/pip install lightgbm==4.6.0 scikit-learn==1.5.2 pandas==2.2.3 numpy==2.1.3
```

---

## 8. Phụ lục GPU — không thực hiện

Không triển khai vLLM/gemma vì cần xin quota GPU (`NCASv3_T4`) và tốn kém. VM không có GPU, không có `nvidia-smi`.

---

## 9. Dọn dẹp (Part 7)

```bash
az group delete --name ai-lab-rg --yes --no-wait
```

Đã xoá toàn bộ resource group `ai-lab-rg` (VM `ai-cpu-node`, VNet `ai-lab-vnet`, NSG `ai-lab-nsg`, public IP `ai-cpu-nodePublicIP`).

| Kiểm tra | Kết quả |
|---|---|
| `az group list` còn `ai-lab-rg`? | Không |
| VM nào còn trong subscription? | Không có |
| Public IP của VM | Đã xoá |
| Port 22 trên IP cũ | Đóng |

Lệnh `--no-wait` trả về ngay, nhưng Azure xoá tài nguyên theo thứ tự phụ thuộc: VM biến mất sau ~1 phút, còn public IP Standard mất thêm ~5 phút mới giải phóng xong. Cần poll lại sau khi xoá để xác nhận tài nguyên thực sự không còn tính phí.

Các resource group còn lại trong subscription (`NetworkWatcherRG`, `defaultresourcegroup-eus`, `MA_defaultazuremonitorworkspace-eus_malaysiawest_managed`) là resource group hệ thống do Azure tự tạo, không phải của lab.

Sau khi dọn dẹp, không còn tài nguyên lab nào phát sinh chi phí.