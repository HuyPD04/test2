Đúng hướng. Reward là phần quyết định agent học hành vi gì, nên cần đọc nó như một hàm tối ưu chứ không chỉ là vài hệ số.

Tôi sẽ dùng đúng nhánh đang chạy: `use_simplified_reward: true` trong [rl.yaml](/D:/RL-SAHI/configs/rl.yaml). Nó tương ứng với Equation (2)–(7) trong thesis và code [slice_env.py](/D:/RL-SAHI/src/rl_sahi/rl/slice_env.py).

## 1. Hai tầng reward

Reward có hai giai đoạn:

```text
Mỗi lần di chuyển/zoom:
  shaped environment reward = r_base

Khi STOP và thật sự chạy detector crop:
  terminal reward = r_base + crop-outcome reward
```

`r_base` dạy agent đi tới ROI tốt về hình học, scale và tính mới.

Crop-outcome reward kiểm tra câu hỏi quan trọng hơn:

> ROI này có thật sự làm detector phát hiện thêm thứ hữu ích không?

---

## 2. Ký hiệu cần biết

Gọi:

- \(a_r\): tỷ lệ diện tích ROI trên toàn ảnh.
- \(\eta\): scale gain, xấp xỉ:

\[
\eta=\frac{\min(H,W)}{\text{side(ROI)}}
\]

ROI càng nhỏ thì \(\eta\) càng lớn, object càng được phóng to khi crop.

- \(c_t\): số hard target mới mà ROI hiện tại phủ được, chưa từng commit trước đó.
- \(c_{t-1}\): số hard target mới mà ROI trước đó phủ được.
- \(n_\text{new}\): số hard target mới được **commit** khi chọn `STOP`.
- \(q_\text{new}\): tổng quality score của các hard target mới.
- \(n_\text{ret}\): số hard target đã được commit từ trước nhưng ROI hiện tại lại phủ tiếp.
- \(o_\text{old}\): overlap với ROI đã accepted.
- \(o_\text{att}\): overlap với ROI đã attempted.
- \(o_\text{det}\): mức ROI phủ lên các detection confidence cao đã có.

Hard target chỉ được tính là “hit” nếu:

1. ROI có diện tích không vượt `0.12`.
2. Scale gain ít nhất `2.5`.
3. Tâm object không sát biên ROI hơn margin `0.08`.
4. Object sau resize có scale hợp lý.

---

## 3. Quality của một hard target

Projected size:

\[
z=
\max(w_\text{obj},h_\text{obj})
\cdot
\frac{320}{\text{side(ROI)}}
\]

Score dạng tam giác:

\[
q(z)=
\operatorname{clip}
\left(
\min
\left[
\frac{z-12}{32-12},
\frac{96-z}{96-32}
\right],
0,1
\right)
\]

Nghĩa là:

| Projected size \(z\) | Quality |
|---:|---:|
| \(z\le12\) | 0 |
| \(z=32\) | 1 |
| \(z\ge96\) | 0 |

Tại sao là `12–32–96`?

- `<12 px`: object sau crop vẫn quá nhỏ.
- `32 px`: đúng ngưỡng COCO bắt đầu gọi là small object; đủ rõ để detector có cơ hội tốt hơn.
- `>96 px`: ROI quá hẹp so với object, rủi ro mất context/cắt object; không còn là mục tiêu small-object enhancement.

`320` ở đây là độ phân giải **đánh giá reward**, không phải detector crop thực tế. Detector crop chạy `512`; đây là heuristic để policy ưu tiên một scale mong muốn ổn định.

---

## 4. Reward khi chưa `STOP`

Với action di chuyển hoặc zoom, phần hard-region reward là:

\[
R_\text{hard}^{\text{move}}
=
0.5\cdot
\operatorname{clip}(c_t-c_{t-1},-4,4)
-
0.25\cdot \mathbf1[c_t=0]
-
0.5\cdot\min(n_\text{ret},4)
\]

### 4.1. Coverage progress: hệ số `0.5`

\[
+0.5\cdot\operatorname{clip}(c_t-c_{t-1},-4,4)
\]

Nếu vừa dịch ROI và che thêm một hard target mới:

\[
c_t-c_{t-1}=1
\Rightarrow +0.5
\]

Nếu rời khỏi một hard target:

\[
c_t-c_{t-1}=-1
\Rightarrow -0.5
\]

Clip ở `[-4, 4]` giới hạn thành:

\[
[-2, +2]
\]

Lý do:

- Có dense feedback trước khi crop.
- Không để ảnh cực đông object tạo reward quá lớn và chi phối train.
- `0.5` nhỏ hơn reward cho `STOP` thành công (`0.75/object`), nên agent được khuyến khích tiến gần mục tiêu nhưng không “đi mãi” chỉ để ăn progress reward.

### 4.2. Empty hard ROI: `-0.25` khi đang di chuyển

Nếu ROI không chứa hard target mới:

\[
-1.0\times0.25=-0.25
\]

Đây là penalty nhẹ. Agent vẫn được phép đi qua vùng tạm thời không có target để tới vị trí tốt hơn.

Tại `STOP`, cùng lỗi này bị phạt `-1.0`, mạnh gấp 4 lần, vì commit một ROI rỗng là đã tiêu tốn lượt crop detector.

### 4.3. Repeated hard target: `-0.5` mỗi target

\[
-0.5\cdot\min(n_\text{ret},4)
\]

Nếu ROI lại phủ một hard target đã commit:

- 1 target cũ: `-0.5`
- 4 target cũ: `-2.0`
- Nhiều hơn 4: vẫn tối đa `-2.0`

Clip tránh việc ảnh có nhiều target cũ tạo penalty khổng lồ. Mục tiêu là nói với agent “vùng này đã được xử lý”, không phải làm mọi reward khác mất ý nghĩa.

---

## 5. Reward khi `STOP`

Khi chọn `STOP` và ROI chứa hard target mới:

\[
R_\text{hard}^{\text{STOP}}
=
0.75n_\text{new}
+
0.75\cdot0.3\cdot
\operatorname{clip}
\left(
q_\text{new}\frac{0.12}{a_r},
0,3
\right)
\]

cộng thêm empty/repeat penalty như trên.

### 5.1. `0.75 × n_new`: thưởng commit thật

Mỗi hard target mới được commit:

\[
+0.75
\]

Tại sao `0.75`, lớn hơn progress reward `0.5`?

Vì coverage trong lúc di chuyển chỉ là dấu hiệu tạm thời. `STOP` là hành động cam kết dùng compute để crop. Reward cao hơn tạo động lực để agent kết thúc ở ROI tốt, thay vì lượn thêm trong vùng target.

### 5.2. Density reward: tối đa `+0.675`

Hệ số:

\[
0.75\times0.3=0.225
\]

Nên:

\[
0.225\cdot
\operatorname{clip}
\left(
q_\text{new}\frac{0.12}{a_r},
0,3
\right)
\]

tối đa là:

\[
0.225\times3=0.675
\]

Với cùng một target quality \(q_\text{new}\):

- ROI diện tích `0.12`: density factor xấp xỉ \(q_\text{new}\).
- ROI diện tích `0.06`: density factor tăng gấp đôi.
- ROI nhỏ hơn nữa: được tăng nhưng bị cap ở `3`.

Ý nghĩa:

> Cùng bắt được một object, ROI nhỏ gọn và zoom đúng được đánh giá tốt hơn ROI lớn.

Tại sao density bonus không lớn hơn reward hard hit?

Vì compactness chỉ là tiêu chí phụ. Agent phải ưu tiên “có target mới” trước; sau đó mới tối ưu ROI gọn.

---

## 6. Efficiency reward: phạt nhẹ ở mọi bước

\[
R_\text{eff}
=
-0.5(0.05+0.5a_r)
\]

Khai triển:

\[
R_\text{eff}=-0.025-0.25a_r
\]

Vì ROI hợp lệ thường có:

\[
a_r\le0.12
\]

penalty nằm gần:

\[
[-0.055,-0.025]
\]

Nó rất nhỏ so với TP reward `+3`, hay overlap penalty `-2/-3`.

Đó là chủ ý: đây là một **tie-breaker**, không phải lực chính.

Nếu hai ROI đều có tiềm năng tương tự, ROI nhỏ/gọn được ưu tiên. Nhưng không nên để agent bỏ qua một hard object chỉ vì ROI đó lớn hơn một chút.

---

## 7. Constraint reward: các điều kiện bị xem là “sai bản chất”

\[
R_\text{con}
=
-3C-o_\text{det}
\]

với:

\[
C=
\left[\frac{a_r}{0.12}-1\right]_+
+
\left[\frac{2.5}{\eta}-1\right]_+
+
\mathbf1[o_\text{old}\ge0.5]
+
\frac23\mathbf1[o_\text{att}\ge0.5]
\]

Trong đó:

\[
[x]_+=\max(x,0)
\]

### 7.1. ROI vượt diện tích tối đa `0.12`

\[
-3\left[\frac{a_r}{0.12}-1\right]_+
\]

Ví dụ ROI có \(a_r=0.18\):

\[
\frac{0.18}{0.12}-1=0.5
\]

\[
R=-3(0.5)=-1.5
\]

ROI lớn gần giống chạy detector toàn ảnh: object không được phóng đủ, compute advantage giảm. Vì thế penalty này mạnh.

### 7.2. Scale gain thấp hơn `2.5`

\[
-3\left[\frac{2.5}{\eta}-1\right]_+
\]

Ví dụ \(\eta=2\):

\[
\frac{2.5}{2}-1=0.25
\]

\[
R=-0.75
\]

Nếu crop chỉ phóng object lên 2 lần, lợi ích small-object detection chưa đủ rõ. `2.5` là ngưỡng “ROI phải nhỏ hơn khoảng 40% cạnh ngắn ảnh”.

### 7.3. Overlap với accepted ROI: `-3`

Nếu:

\[
o_\text{old}\ge0.5
\]

thì:

\[
R=-3
\]

Overlap ở đây đo phần ROI hiện tại bị vùng lịch sử chiếm. Từ 50% trở lên, phần lớn crop có nguy cơ redundant.

Penalty này rất mạnh vì accepted ROI đã đóng góp detection vào kết quả cuối; crop lại cùng vùng gần như chắc chắn tạo duplicate.

### 7.4. Overlap với attempted ROI: `-2`

\[
-3\times\frac23=-2
\]

Attempted ROI bị phạt nhẹ hơn accepted ROI:

- Accepted ROI: đã chứng minh hữu ích, crop lại là lãng phí rõ ràng.
- Attempted ROI: có thể từng bị reject do detection/gate, nên về lý thuyết còn một ít khả năng ROI lân cận tốt hơn.

Nhưng vẫn cần `-2` để agent không lặp lại failure gần như y hệt.

### 7.5. Overlap với full-image detection confidence cao: tối đa `-1`

\[
-o_\text{det}
\]

\[
o_\text{det}
=
\operatorname{clip}
\left(
\frac1{3}
\sum_{\text{score}\ge0.5}
\frac{\operatorname{intersection}(\text{ROI},\text{box})}
{\operatorname{area}(\text{box})},
0,1
\right)
\]

Nó dùng intersection-over-detection-area, không dùng IoU. Nếu ROI bao trọn một box confidence cao, object đó đã được detector toàn ảnh giải thích tốt; crop lại ít giá trị.

Chia cho `3` nghĩa là khoảng ba detection confidence cao bị ROI phủ mạnh mới đạt penalty tối đa `-1`.

---

## 8. Stop bonus và stop penalty

Ngoài \(R_\text{hard}\), `STOP` có thêm:

\[
R_\text{STOP}=
\begin{cases}
+0.5\min(q_\text{new},4),
&n_\text{new}>0,\;o_\text{old}<0.5\\
+0.5\cdot0.3\min(z,2),
&\text{không có hard GT và }z>0.3\\
-0.5\cdot0.5,
&\text{còn lại}
\end{cases}
\]

### Hard target branch

\[
+0.5\min(q_\text{new},4)
\]

Tối đa:

\[
+2.0
\]

Nó không chỉ thưởng số object, mà thưởng chất lượng scale/context của object. Một ROI chạm được object nhưng phóng nó không đúng scale có \(q\) thấp, nên không được bonus nhiều.

### Observable branch

Khi không có hard GT, source dùng observable score \(z\), lấy từ:

- Proposal confidence thấp.
- Small uncertain detection.
- Objectness map.
- Density trong ROI.

Bonus tối đa:

\[
0.5\times0.3\times2=0.3
\]

Nó nhỏ hơn hard-GT reward vì detector evidence không đáng tin bằng supervision GT.

### Stop sớm sai: `-0.25`

\[
-0.5\times0.5=-0.25
\]

Đây là penalty vừa phải. Nó không làm một `STOP` chưa tốt trở thành thảm họa ngay lập tức; crop-outcome reward phía sau mới là phán quyết mạnh hơn.

---

## 9. \(r_\text{base}\) trong code hiện tại

Tóm lại:

\[
r_\text{base}
=
R_\text{hard}
+
R_\text{eff}
+
R_\text{con}
+
R_\text{STOP}
\]

Ví dụ một ROI `STOP` tốt:

- Có 1 hard target mới, \(q_\text{new}=0.8\).
- ROI area ratio \(a_r=0.06\).
- Không overlap.
- Không phủ detection confidence cao.

Ta có:

\[
R_\text{hard}=0.75
+
0.225\cdot
\left(0.8\frac{0.12}{0.06}\right)
=
0.75+0.36
=
1.11
\]

\[
R_\text{eff}
=
-0.5(0.05+0.5\cdot0.06)
=
-0.04
\]

\[
R_\text{con}=0
\]

\[
R_\text{STOP}=0.5(0.8)=0.4
\]

Vậy:

\[
r_\text{base}=1.11-0.04+0+0.4=1.47
\]

ROI này tốt về mặt hình học, nhưng chưa biết crop detector có thật sự tốt hay không.

---

## 10. Crop-outcome reward: phần semantic quan trọng nhất

Sau crop 512 và merge, thesis/source dùng:

\[
R_\text{crop}
=
0.5U_\text{new}
+
3.0[\Delta TP]_+
-
0.5[\Delta FP]_+
-
1.2\mathbf1[m=0]
-
1.2\mathbf1[m>0\land\Delta_\text{new}=0]
\]

Trong đó:

- \(U_\text{new}\): tổng confidence của detection mới sau merge.
- \(\Delta TP\): true positive tăng thêm.
- \(\Delta FP\): false positive tăng thêm.
- \(m\): số detection hợp lệ trong crop.
- \(\Delta_\text{new}\): số detection thật sự mới.

Hai penalty cuối loại trừ nhau:

- Crop rỗng: `-1.2`.
- Crop có box nhưng tất cả duplicate/vô ích: `-1.2`.

### Vì sao TP là `+3`, novelty chỉ `+0.5`?

Một detection confidence cao nhưng là FP vẫn không được xem là thành công. Do đó:

- Novelty utility: tín hiệu mềm, tối đa tăng dần theo confidence.
- TP gain: tín hiệu đúng semantic, được ưu tiên mạnh.

Một TP mới score `0.7`, không FP:

\[
R_\text{crop}
=
0.5(0.7)+3
=
3.35
\]

Nếu crop chỉ tạo một FP score cao `0.9`:

\[
R_\text{crop}
=
0.5(0.9)-0.5
=
-0.05
\]

Nếu crop có box nhưng không mới:

\[
R_\text{crop}=-1.2
\]

Như vậy policy không thể kiếm reward lâu dài chỉ bằng việc làm detector sinh thêm box.

---

## 11. Gate accepted/rejected

Crop chỉ accepted nếu:

\[
\Delta_\text{new}\ge1
\]

\[
s_\text{max}\ge0.1
\]

và:

\[
\Delta_\text{new}=1
\quad\lor\quad
U_\text{new}\ge0.3
\]

Ý nghĩa:

- Có ít nhất một detection mới.
- Detection mới phải có confidence tối thiểu `0.1`.
- Nếu chỉ có đúng một detection mới, một detection đủ confidence là có thể giữ.
- Nếu crop sinh nhiều box, tổng utility phải đạt `0.3`; tránh nhận một cụm box rất yếu.

`0.1` khá thấp, có chủ đích: object nhỏ sau crop có thể vẫn confidence thấp nhưng đáng giữ để tăng recall. Novelty/NMS/TP-FP reward sẽ kiểm soát phần còn lại.

---

## 12. Terminal reward cuối cùng

Trong trainer, [batched_trainer.py](/D:/RL-SAHI/src/rl_sahi/rl/batched_trainer.py:165), nếu crop accepted:

\[
r_T
=
r_\text{base}
+
0.5n_\text{new}
+
R_\text{crop}
-
0.25
\]

Nếu crop rejected:

\[
r_T
=
\min(r_\text{base},0)
+
\min(R_\text{crop},0)
-
0.5
-
0.25
\]

### 12.1. Accepted case

Tiếp ví dụ trên, crop có một TP mới score `0.7`:

\[
r_T=
1.47+0.5+3.35-0.25=5.07
\]

Đây là reward rất rõ: ROI đúng về geometry, đúng về scale, và detector thật sự có thêm TP.

### 12.2. Rejected case

Giả sử ROI hình học trông tốt nên:

\[
r_\text{base}=1.47
\]

nhưng crop chỉ sinh duplicate box:

\[
R_\text{crop}=-1.2
\]

Thì:

\[
r_T
=
\min(1.47,0)
+
\min(-1.2,0)
-
0.5-0.25
=
-1.95
\]

Đây là logic rất quan trọng:

> Reward geometry dương không được phép “che” việc crop thực tế vô ích.

Nói cách khác, GT/hard-target reward chỉ là guidance. Kết quả crop mới là kiểm chứng cuối.

---

## 13. Thứ tự mạnh/yếu của các hệ số

| Thành phần | Biên độ gần đúng | Ý nghĩa |
|---|---:|---|
| TP mới | `+3` / TP | Mục tiêu semantic mạnh nhất |
| Hard target commit | `+0.75` / target | Dạy nơi nên crop |
| Stop quality | tối đa `+2` | Ưu tiên ROI đúng scale/context |
| Non-stop progress | tối đa `±2` | La bàn điều hướng |
| Accepted hard hit | `+0.5` / target | Xác nhận ROI hữu ích |
| Empty/no-gain crop | `-1.2` | Crop vô ích phải bị nhớ |
| Accepted overlap | `-3` | Tránh crop trùng kết quả đã giữ |
| Attempted overlap | `-2` | Tránh lặp lại failure |
| Rejected crop | `-0.5` | Phạt quyết định commit sai |
| Crop cost | `-0.25` | Kiểm soát compute |
| Efficiency cost | khoảng `-0.025` đến `-0.055` | Tie-break ROI nhỏ |

Thứ bậc này cho thấy ưu tiên thiết kế:

```text
TP mới
  > hard target đúng
    > ROI compact, đúng scale
      > tiến triển khi điều hướng
        > chi phí compute nhỏ
```

---

## 14. Một chi tiết code quan trọng

Vì `use_simplified_reward=true`, một số field trong config **không tham gia reward đang chạy**, ví dụ:

- `step_penalty`
- `area_penalty`
- `large_roi_penalty`
- `low_scale_penalty`
- `stop_target_reward`
- `stop_early_penalty`

Chúng thuộc `_legacy_reward()`, không phải `_simplified_reward()` hiện tại.

Do đó, khi tuning reward, nên ưu tiên các số thực sự active:

```text
target_reward
hard_coverage_progress_reward
empty_hard_penalty
repeated_hard_penalty
efficiency_weight
constraint_weight
attempted_overlap_penalty
detected_overlap_penalty
stop_bonus_weight
crop_* reward / penalty
hard_hit_reward
rejected_crop_penalty
crop_attempt_penalty
```

Cuối cùng, reward được clip trong khoảng `[-10, 10]` trước khi tạo target học DQN. Điều này bảo vệ Q-learning nếu một crop chứa nhiều TP/hard target và raw terminal reward quá lớn.