# Báo cáo tổng hợp point-cloud grounding

Tài liệu độc lập theo cấu trúc paper, dành cho người chưa biết lịch sử experiment.

- report.pdf: bản chia sẻ/in.
- report.html: bản đọc offline với công thức và hình.
- report.md: nội dung nguồn để chỉnh sửa.
- figures/: sơ đồ kiến trúc, kết quả tổng hợp và ví dụ RGB/GT/prediction; PNG + SVG.
- evidence.json: số liệu đầy đủ, nguồn/checkpoint/hash và quy tắc chọn ví dụ.

Data: E157, 1024 train / 128 validation / 5×128 heldout scenes; 4096 input / 128 output points, 4 tasks. Cấu hình hoàn tất: SAM3 (E175,50k/best12k,FP32), DINO block6 (E177,30k/best11k,BF16) và block24 (E178,30k/best12k,BF16). Một EMD loss, không thêm SE(3)/scale augment; visual backbones frozen, encoder/decoder/head train từ đầu.

Test: validation-best checkpoint cố định; canonical splits, 4 coordinate SE(3) views, 4 novel shape families, 6 physical camera conditions và feature-correspondence controls. Full metrics trong evidence.json. Không có training/inference mới; không hợp nhất best-per-test thành một model chưa đánh giá. Shape/size/visibility, một seed và các confound SAM3/DINO được ghi rõ trong report.

Kiểm chứng kết quả nguồn: SAM3 đã kiểm tra raw GT/source/checkpoint, EMD/centroid, coverage spot-checks và toàn bộ SE(3) drift; DINO-B6 audit34816 records; DINO-B24 kiểm tra hash/finiteness34816records và tính lại độc lậpEMD5632bestIID/OODqueries. Không tuyên bố đã độc lập tính lại toàn bộ coverage DINO. document_check.json ghi kiểm tra hiển thị/PDF, tách biệt kiểm chứng metrics.

Final document check: 11-page PDF (~1.4MB),11tables,3figures,4equations; text/Unicode extractable. All23sourcefilehashes rechecked; HTML/PDF/images HTTP200 and browser has no pageerrors. PDF and figure layout visually reviewed. See document_check.json. Read online: http://localhost:17070/pointcloud_report/report.html .
