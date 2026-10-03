# LAB 16 — Cloud AI Environment Setup (Azure track) — Báo cáo kết quả

**Ngày chạy:** 2026-10-03 · **Subscription:** Azure for Students (ID đã redact)
**Resource group:** `ai-lab-rg` (malaysiawest) · **VM:** `ai-cpu-node` — `Standard_B2s_v2`
**Public IP:** `<vm-ip>` · **OS:** Ubuntu 24.04 LTS (kernel 6.17.0-1022-azure)

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
| Inference throughput (1000 rows) | **93,574 rows/s** (10.687 ms / batch 1000) |

Thời gian cross-validation để chọn `n_estimators`: 40.17 s (được tính riêng, **không** cộng vào "training time").

Tại ngưỡng F1 tối ưu (0.95): F1 = 0.2334, Precision = 0.1341, Recall = 0.8980.

---

## 2. Ba phát hiện quan trọng khi chạy trên dataset này

Trong quá trình làm lab, kết quả ban đầu **sai rõ rệt** (AUC chỉ 0.936, `Best iteration = 1`, F1 = 0.166). Ba nguyên nhân đã được kiểm chứng và xử lý:

**(1) `Time` phải bị loại bỏ.** Đây là timestamp thô, không mang tín hiệu dự đoán. 5-fold CV: **0.897 không có** `Time` vs **0.880 có** `Time`.

**(2) `is_unbalance=True` là bắt buộc.** Dataset chỉ có **0.1727%** gian lận (492/284.807). Không cân bằng class, LightGBM học trên ~400 mẫu dương và overfit rất nhanh. Cột này chính là nguyên nhân khiến `Best iteration = 1`.

**(3) Early stopping trên một validation split nhỏ bị hỏng — đây là lỗi quan trọng nhất.** Validation split chỉ chứa **~79 mẫu gian lận**, nên AUC trên split đó gần như ngẫu nhiên:

| n_estimators | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| AUC (không cân bằng) | 0.9415 | **0.6855** | 0.7771 | 0.7880 | 0.8203 |
| AUC (`is_unbalance`) | 0.9113 | **0.2498** | 0.8974 | 0.8862 | 0.8854 |
| AUC (`scale_pos_weight=100`) | 0.9590 | **0.1133** | 0.9056 | 0.9348 | 0.9183 |

AUC **sụp xuống dưới 0.5 ở vòng 2** dưới mọi cách cân bằng, và điều này lặp lại **giống hệt nhau trên LightGBM 4.6.0 lẫn 4.7.0** (đã kiểm tra để loại trừ khả năng đây là bug của thư viện — và nó **không phải** bug). Do đó validation AUC đạt đỉnh giả ở vòng 1, early stopping dừng ngay, và `predict_proba` chỉ dùng **1 cây**.

Ngoài ra, **AUC do chính LightGBM ghi vào lịch sử eval cũng không đáng tin** trên dataset này: đường cong bị đóng băng ở đúng một giá trị từ vòng ~450 trở đi. Vì vậy benchmark **tự tính AUC bằng sklearn trên dự đoán thật**, mỗi fold chỉ fit 1 lần rồi dùng `num_iteration` để lấy dự đoán tại từng mốc.

Kết quả sweep (5-fold, sklearn AUC) — đường cong tăng đều rồi đi ngang, nên lựa chọn 400 là có cơ sở:

| n_estimators | 50 | 100 | 200 | 300 | **400** | 500 | 600 |
|---|---|---|---|---|---|---|---|
| Mean CV-AUC | 0.8732 | 0.8758 | 0.8804 | 0.8811 | **0.8875** | 0.8875 | 0.8875 |

---

## 3. Về F1 / Precision thấp — đây là hệ quả, không phải lỗi

Precision = 0.119 và F1 = 0.211 trong khi AUC = 0.940 là **mâu thuẫn về mặt trực giác nhưng hoàn toàn đúng**. Với chỉ 394 mẫu gian lận trong tập huấn luyện và ngưỡng cắt cố định 0.5, mô hình ưu tiên **bỏ sót ít** hơn là **báo động giả ít**: Recall = 0.898 nhưng Precision thấp.

`is_unbalance=True` đẩy toàn bộ điểm dự đoán lên cao, nên 0.5 vốn đã là một ngưỡng vận hành tệ. Ngưỡng tối ưu F1 tìm được là **0.95** (F1 = 0.2334). Muốn cải thiện đáng kể cần thu thập thêm dữ liệu gian lận hoặc kỹ thuật resampling (SMOTE) — nhưng SMOTE **không** dùng ở đây vì làm sai lệch thật trên tập test.

---

## 4. Tài nguyên & chi phí (Part 5)

**CPU** — `Intel Xeon Platinum 8370C @ 2.80GHz`, 2 vCPU (1 core × 2 thread), cấp hình khác README một chút so với `Standard_B2s` (2 vCPU / 4 GB) vì VM thực tế là **`Standard_B2s_v2`**.

- Lúc idle: **0.5 – 7.6%**
- Lúc chạy `benchmark.py`: **95–100%** cả 2 vCPU (`top` ghi nhận tiến trình `python` dùng **200.0% CPU**, tức đã dùng hết 2 core)
- Azure Monitor (PT5M): **peak 73.6%**, mean 29.8%

**RAM** — 7.8 GiB tổng; khi train chỉ dùng ~1.25 GiB (~16%), còn **6.7–6.9 GiB available**. Dataset 284k×29 dễ dàng nằm gọn trong RAM, không cần swap.

**Network** — `ip -s link` trên `eth0`: **RX ~845 MB** (chủ yếu là tải dataset 143.8 MB + cài đặt package), **TX ~11 MB**. 0 error, 0 dropped packet. Đúng như thiết kế: NSG chỉ mở port 22 từ một IP duy nhất (`<my-public-ip>/32`, đã redact), không mở cổng vLLM 8000 vì không làm phụ lục GPU.

**Disk** — `/dev/root` 29 GB, đã dùng 4.7 GB (17%).

**Chi phí** — Cost Management API (`ActualCost`, month-to-date, lọc theo resource group):

| Ngày | PreTaxCost |
|---|---|
| 2026-10-02 | 0.224831 USD |
| 2026-10-03 | 0.108861 USD |
| **Tổng** | **0.333692 USD** |

Khớp với ước tính ~$0.05/giờ trong README (vùng Malaysia West thay cho East US).

---

## 5. Bảo mật — đã áp dụng least-privilege

```
Name               Prio  Access  Source              DestPort  Protocol
allow-ssh-from-me  100   Allow   <my-public-ip>/32   22        Tcp
```

Chỉ **một** rule trong NSG: SSH port 22, giới hạn đúng `/32` của IP cá nhân. Không có rule nào mở `0.0.0.0/0`, không mở port 8000. Public IP là **Static** (Standard SKU).

> Ghi chú: README khuyến nghị Public IP Standard (~$0.005/giờ). VM này dùng **Static** IP nên **không** phát sinh phí IP theo giờ — chi phí thực tế chỉ đến từ VM.

---

## 6. Deliverables (Part 6)

| # | Deliverable | File |
|---|---|---|
| 1 | Screenshot / output terminal đầy đủ | `benchmark_output.txt` |
| 2 | File kết quả | `benchmark_result.json` |
| 3 | Resource usage | `resource_usage.txt`, `resource_usage_under_load.txt`, `azure_monitor_metrics.txt` |
| 4 | Cost Management | `cost_data.txt` (+ Portal screenshot nếu giảng viên yêu cầu) |
| 5 | cloud-init đã dùng | `cloud-init-cpu.yaml` |
| 6 | Báo cáo nhận xét | `REPORT.md` (file này) |
| — | Script benchmark | `benchmark.py` |
| — | Hạ tầng đã dựng | `infra_summary.txt` |

---

## 7. Sai lệch so với README cần lưu ý

| README ghi | Thực tế |
|---|---|
| VM `Standard_B2s` | **`Standard_B2s_v2`** (4 GB → 8 GB) |
| Location `eastus` | **`malaysiawest`** (giá và quota khác) |
| `pip3 install ... kaggle` → CLI `kaggle` dùng được | CLI **không** có trong PATH; `/usr/local/bin/kaggle` là symlink hỏng trỏ tới `/opt/ml-env/bin/kaggle` (venv rỗng). Phải dùng `~/ml-env/bin/python` sau khi tự tạo venv. |
| Ước tính giá theo East US | Đã thay bằng số liệu thực tế Malaysia West (§4) |
| Early stopping → "Best iteration" | **Không dùng** — xem §2(3). `n_estimators` chọn bằng 5-fold CV. |

Về môi trường: cloud-init `pip3 install` trên Ubuntu 24.04 đã cài package vào `~/.local` (PEP 668), còn `kaggle` CLI cần venv riêng. Để tái lập được, tạo venv với phiên bản ghim:

```bash
python3 -m venv ~/ml-env
~/ml-env/bin/pip install lightgbm==4.6.0 scikit-learn==1.5.2 pandas==2.2.3 numpy==2.1.3
```

---

## 8. Phụ lục GPU (tuỳ chọn) — KHÔNG thực hiện

Không triển khai vLLM/gemma vì cần xin quota GPU (`NCASv3_T4`) và tốn kém. VM hiện tại không có GPU; không có `nvidia-smi`.

---

## 9. Dọn dẹp (Part 7) — ĐÃ THỰC HIỆN

```bash
az group delete --name ai-lab-rg --yes --no-wait
```

Đã xoá **toàn bộ** resource group `ai-lab-rg` (VM `ai-cpu-node`, VNet `ai-lab-vnet`, NSG `ai-lab-nsg`, public IP `ai-cpu-nodePublicIP`).

**Đã xác minh sau khi xoá:**

| Kiểm tra | Kết quả |
|---|---|
| `az group list` còn `ai-lab-rg`? | Không — đã xoá |
| VM nào còn trong subscription? | Không có |
| Public IP `<vm-ip>` | Đã xoá |
| Port 22 trên IP cũ | Đóng (không còn phản hồi) |

> **Lưu ý về thời gian hoàn tất:** lệnh `--no-wait` trả về ngay, nhưng Azure xoá tài nguyên theo thứ tự phụ thuộc. VM biến mất sau ~1 phút, còn public IP Standard mất **thêm ~5 phút** mới giải phóng xong — đây là lý do cần poll lại sau khi xoá, nếu không sẽ tưởng còn tài nguyên tính phí. Resource group chỉ biến mất khỏi danh sách sau khi mọi tài nguyên con đã xoá xong.

Các resource group còn lại trong subscription (`NetworkWatcherRG`, `defaultresourcegroup-eus`, `MA_defaultazuremonitorworkspace-eus_malaysiawest_managed`) là resource group hệ thống do Azure tự tạo, **không phải** của lab và không do ta tạo ra.

**Không còn tài nguyên lab nào tính phí.**