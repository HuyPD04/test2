DQN là phần “não quyết định” của hệ thống. Nó không dự đoán box, không dự đoán class, cũng không dự đoán tọa độ ROI trực tiếp. Nó trả lời đúng một câu:

> Trong state hiện tại, hành động ROI nào sẽ đem lại tổng reward tương lai lớn nhất?

Source hiện dùng tổ hợp:

```text
Dueling Double DQN
+ action masking
+ 3-step return
+ prioritized replay
+ soft target network
```

## 1. DQN học hàm gì?

Mạng học:

\[
Q_\theta(s,a)
\]

Trong đó:

- \(s\): state 3,612 chiều.
- \(a\): một trong 11 action.
- \(Q\): tổng reward kỳ vọng nếu chọn \(a\), rồi tiếp tục hành xử tốt.

Ví dụ:

\[
Q(s,\text{ZOOM\_IN})=2.4
\]

không có nghĩa zoom in sẽ nhận ngay reward `2.4`. Nó nghĩa là:

> Nếu zoom in ở ROI này, rồi tiếp tục dịch/zoom/STOP hợp lý, tổng reward discounted dự kiến là 2.4.

Mạng cho ra đồng thời 11 Q-value:

\[
Q_\theta(s)=
[
Q(s,a_1),Q(s,a_2),\dots,Q(s,a_{11})
]
\]

Không có softmax. Q-value không phải probability và không cần tổng bằng 1. Nó có thể âm hoặc dương.

---

## 2. Bellman equation: nguồn gốc target của DQN

DQN dựa trên Bellman optimality equation:

\[
Q^*(s_t,a_t)
=
\mathbb E
\left[
r_t+
\gamma
\max_{a'}Q^*(s_{t+1},a')
\right]
\]

Diễn giải:

```text
Giá trị action hiện tại
=
reward nhận ngay
+
giá trị tốt nhất có thể đạt được từ state kế tiếp
```

Ví dụ:

```text
State hiện tại: ROI chưa tối ưu
Action: RIGHT
Reward hiện tại: +0.5 vì phủ thêm hard target
State sau: ROI gần vùng đáng crop hơn
Best future Q: +3.0
```

Nếu \(\gamma=0.95\):

\[
Q(s,\text{RIGHT})
\approx
0.5+0.95(3.0)=3.35
\]

Điểm mạnh của Bellman learning là agent không cần nhãn “action đúng”. Nó tự tạo target từ reward thật và dự đoán về tương lai.

---

## 3. DQN source thực sự không dùng one-step target

Replay của source lưu n-step return với \(n=3\):

\[
R_t^{(3)}
=
r_t+\gamma r_{t+1}+\gamma^2r_{t+2}
\]

Vì vậy target thực tế:

\[
y_t=
R_t^{(3)}
+
\gamma^3(1-d_t)
Q(s_{t+3},a^*)
\]

Trong config:

\[
\gamma=0.95
\]

nên:

\[
\gamma^3=0.857375
\]

Trong hàm `optimize()`, biến `gamma` đã được caller truyền vào dưới dạng \(\gamma^3\), không phải \(0.95\) đơn lẻ.

Tại terminal:

\[
d_t=1
\]

nên:

\[
y_t=R_t^{(3)}
\]

Không bootstrap từ state sau terminal, vì rollout đã kết thúc tại `STOP`/forced termination.

---

## 4. Vì sao DQN dùng Q-value thay vì dự đoán ROI trực tiếp?

Có hai cách tiếp cận.

### Cách liên tục

Dự đoán trực tiếp:

\[
(c_x,c_y,w,h)
\]

Cách này cần policy gradient hoặc actor-critic, như PPO/SAC. Nhưng nó khó hơn vì action space liên tục, ROI rất nhạy với scale, cần xử lý boundary và overlap bằng constraint phức tạp.

### Cách source đang dùng

Dự đoán 11 action rời rạc:

```text
LEFT, RIGHT, UP, DOWN
4 hướng chéo
ZOOM_IN, ZOOM_OUT
STOP
```

Mỗi action là một transformation có kiểm soát.

Điều này biến bài toán từ:

> “Dự đoán chính xác bốn số ROI”

thành:

> “Trong 11 bước hợp lệ, bước nào tăng giá trị dài hạn nhất?”

DQN phù hợp hơn cho không gian hành động rời rạc, nhỏ và có action mask.

---

## 5. Kiến trúc Q-network trong source

Mạng trong [network.py](/D:/RL-SAHI/src/rl_sahi/rl/network.py) tách state thành hai phần.

```text
State 3,612 chiều
│
├── 256 global YOLO features + 28 summary scalars
│      = 284 vector values
│      ↓ MLP
│      512 features
│
└── 13 × 16 × 16 spatial maps
       ↓ Conv 3×3: 13 → 32
       ↓ ReLU
       ↓ Conv 3×3: 32 → 64
       ↓ ReLU
       ↓ AvgPool 4×4
       ↓ Flatten
       64 × 4 × 4 = 1,024 features

concat: 512 + 1,024 = 1,536
       ↓ Linear + ReLU
       512 features
       ↓ Dueling heads
```

### Vì sao spatial maps đi qua CNN?

Spatial map có cấu trúc vị trí:

```text
proposal nằm góc phải trên ROI
history nằm phía trái
current ROI nằm giữa
objectness cluster nằm cạnh ROI
```

CNN học được quan hệ như:

> Nếu objectness/proposal nằm ngay hướng phải-trên so với current ROI, `UP_RIGHT` hoặc `ZOOM_IN` có thể có giá trị cao.

### Vì sao global feature và summary không đi qua CNN?

Global feature và summary là vector có ý nghĩa theo từng chiều:

- Detection count.
- Mean confidence.
- ROI area.
- Step ratio.
- Feature semantic YOLO.

Không tồn tại quan hệ không gian “pixel gần nhau” giữa các chiều này. Dùng MLP là hợp lý hơn.

---

## 6. Dueling DQN: mạng không học trực tiếp 11 Q độc lập

Sau trunk 512 chiều, mạng sinh:

\[
V(s)\in\mathbb R
\]

và:

\[
A(s,a)\in\mathbb R^{11}
\]

Sau đó:

\[
Q(s,a)
=
V(s)+A(s,a)
-\frac1{11}\sum_{a'}A(s,a')
\]

Trong code, average advantage được tính trên cả 11 action head.

### Vai trò của \(V(s)\)

\[
V(s)
\]

đánh giá “state này nhìn chung có tiềm năng không?”

Ví dụ:

```text
ROI ở nền trống, không proposal, không objectness:
V(s) thấp

ROI có nhiều small uncertain proposal, chưa thử:
V(s) cao
```

### Vai trò của \(A(s,a)\)

\[
A(s,a)
\]

đánh giá action này tốt hơn/trở hơn trung bình bao nhiêu.

Ví dụ:

```text
ROI tốt nhưng proposal ở sát mép phải:
A(RIGHT) cao
A(LEFT) thấp
A(STOP) có thể trung bình
```

### Tại sao cần trừ mean advantage?

Nếu chỉ dùng:

\[
Q=V+A
\]

thì có vô số cách cho cùng Q:

```text
V = 10, A = -7
V = 20, A = -17
```

Đều cho \(Q=3\). Mạng khó phân biệt phần nào là “giá trị state”, phần nào là “lợi thế action”.

Phép chuẩn hóa:

\[
A-\overline A
\]

ép trung bình advantage bằng 0. Khi đó \(V(s)\) có vai trò rõ ràng hơn.

---

## 7. Action mask đi vào DQN ở đâu?

Action mask được dùng ở hai nơi quan trọng.

### Khi chọn action

\[
a_t=
\arg\max_{a:m_t(a)=1}Q_\theta(s_t,a)
\]

Code thay Q của action không hợp lệ bằng:

\[
-\infty
\]

rồi mới lấy `argmax`.

### Khi tạo Bellman target

\[
a^*=
\arg\max_{a:m_{t+3}(a)=1}
Q_\theta(s_{t+3},a)
\]

Nếu không mask trong target, DQN có thể học target dựa vào action không thể thực hiện, chẳng hạn:

```text
LEFT khi ROI đã ở mép trái
ZOOM_IN khi ROI đã nhỏ nhất
STOP khi ROI overlap nặng nhưng vẫn có thể thoát
```

Đó sẽ là Q-value của một “tương lai không tồn tại”.

Source có safeguard: nếu một state kế tiếp tình cờ không còn action hợp lệ nào, nó bật `STOP` để tránh `argmax` trên toàn `-∞`.

---

## 8. Double DQN: source chọn action bằng online network, chấm bằng target network

Source dùng:

\[
a^*=
\arg\max_{a:m(a)=1}Q_\theta(s',a)
\]

\[
y=
R^{(3)}
+
\gamma^3(1-d)
Q_{\theta^-}(s',a^*)
\]

Trong đó:

- \(Q_\theta\): online/policy network.
- \(Q_{\theta^-}\): target network.

### DQN thường sẽ làm gì?

DQN thường:

\[
y=
r+\gamma\max_aQ_{\theta^-}(s',a)
\]

Cùng target network vừa chọn action, vừa đánh giá action.

Khi Q-value có nhiễu:

\[
\widehat Q(a)=Q(a)+\epsilon_a
\]

thì:

\[
\mathbb E[\max_a\widehat Q(a)]
\ge\max_aQ(a)
\]

Action có nhiễu dương mạnh thường được chọn, tạo overestimation.

### Double DQN giảm lỗi này thế nào?

- Online network chọn action có vẻ tốt nhất.
- Target network độc lập tương đối sẽ chấm action ấy.

Nếu online network “ảo tưởng” rằng `ZOOM_IN` tốt, target network có cơ hội chấm thấp hơn. Nhờ vậy Q-value bớt phình lớn do nhiễu.

---

## 9. Target network không học trực tiếp bằng gradient

Khi tạo target:

```python
with torch.no_grad():
    target_q = reward + gamma_n * next_q * (1 - done)
```

Gradient chỉ đi vào:

\[
Q_\theta(s_t,a_t)
\]

không đi vào:

\[
Q_{\theta^-}(s_{t+3},a^*)
\]

Nếu gradient đi vào cả hai vế, network vừa thay đổi prediction hiện tại, vừa thay đổi đáp án đang cố khớp; objective trở nên không ổn định.

Sau khi update online network, target network theo sau chậm:

\[
\theta^-
\leftarrow
0.005\theta+0.995\theta^-
\]

Target lag tạo một mục tiêu gần cố định trong vài update, giúp Bellman regression ổn định hơn.

---

## 10. Một update DQN diễn ra thế nào?

Một sample từ replay gồm gần đúng:

\[
(s_t,a_t,R_t^{(3)},s_{t+3},d_t,m_{t+3})
\]

### Bước 1: Q hiện tại của action đã thực hiện

Mạng output 11 Q-value:

\[
Q_\theta(s_t)
\in\mathbb R^{11}
\]

Sau đó `gather` đúng action đã làm:

\[
q_t=Q_\theta(s_t,a_t)
\]

Nếu action lúc đó là `UP_RIGHT`, loss chỉ update trực tiếp Q-head của `UP_RIGHT`, nhưng trunk/shared feature vẫn học từ error đó.

### Bước 2: chọn best next action

\[
a^*=
\arg\max_{a:m(a)=1}Q_\theta(s_{t+3},a)
\]

### Bước 3: target network đánh giá action này

\[
q_\text{next}
=
Q_{\theta^-}(s_{t+3},a^*)
\]

### Bước 4: tạo target

\[
y_t=
R_t^{(3)}
+
0.857375(1-d_t)q_\text{next}
\]

### Bước 5: TD error

\[
\delta_t=q_t-y_t
\]

Ví dụ:

\[
q_t=1.2,\qquad
R_t^{(3)}=2.5,\qquad
q_\text{next}=1.0,\qquad
d_t=0
\]

\[
y_t=2.5+0.857375(1.0)=3.357
\]

\[
\delta_t=1.2-3.357=-2.157
\]

Q hiện tại đánh giá action quá thấp; gradient sẽ tăng \(Q_\theta(s_t,a_t)\).

Nếu transition terminal:

\[
d_t=1
\]

\[
y_t=R_t^{(3)}
\]

Không còn giá trị tương lai.

---

## 11. Loss: weighted Huber loss

Nếu dùng PER:

\[
\mathcal L=
\frac1B
\sum_i
w_i
\operatorname{Huber}(\delta_i)
\]

với:

\[
\operatorname{Huber}(\delta)=
\begin{cases}
\frac12\delta^2,&|\delta|\le1\\
|\delta|-\frac12,&|\delta|>1
\end{cases}
\]

Huber có hai vùng:

- Error nhỏ: giống MSE, tối ưu mượt và chính xác.
- Error lớn: giống L1, không để một reward crop bất thường làm gradient bùng nổ.

Source còn clip gradient norm:

\[
\|\nabla_\theta\mathcal L\|\le10
\]

Và clip reward khi tính target:

\[
R\leftarrow\operatorname{clip}(R,-10,10)
\]

Cả hai cơ chế đều chống instability của Q-learning.

---

## 12. PER kết hợp với DQN như thế nào?

Sau khi tính TD error:

\[
\delta_i=Q_\theta(s_i,a_i)-y_i
\]

source cập nhật priority:

\[
p_i=|\delta_i|+\varepsilon
\]

và sample với:

\[
P(i)=
\frac{p_i^{0.6}}
{\sum_j p_j^{0.6}}
\]

Transition nào DQN đang sai nhiều sẽ được thấy lại nhiều hơn.

Trong bài toán này, các sample giàu thông tin thường là:

```text
STOP đúng, crop cứu được TP mới
STOP sai, crop empty
ROI duplicate bị reject
ROI overlap nặng
ROI đi từ proposal tới hard target đúng scale
```

Nếu sample đều, chúng bị chìm trong nhiều action di chuyển nền/background reward nhỏ.

PER tạo bias, nên source dùng weight:

\[
w_i=(NP(i))^{-\beta}
\]

với \(\beta\) tăng từ `0.4` lên `1.0`. Weight này đưa loss dần về gần phân phối unbiased hơn ở cuối training.

---

## 13. Epsilon-greedy không phải là DQN, nhưng tạo dữ liệu cho DQN

DQN chỉ học được từ action đã xảy ra. Đầu train, Q-values gần random, nên source không thể chỉ dùng:

\[
\arg\max_aQ_\theta(s,a)
\]

Nó dùng:

```text
guided action
→ random valid action
→ greedy Q action
```

Epsilon:

\[
1.0\rightarrow0.05
\]

Guidance probability:

\[
0.25\rightarrow0.05
\]

Guided heuristic tạo các trajectory có proposal/objectness tốt; random valid action khám phá; greedy action khai thác những gì DQN đã học.

Dữ liệu thu được mới đi vào replay để DQN học offline theo batch.

---

## 14. DQN đang học “policy” gián tiếp

Không có policy network riêng:

\[
\pi_\theta(a\mid s)
\]

như PPO.

Policy được suy ra từ Q-network:

\[
\pi(s)
=
\arg\max_{a:m(a)=1}Q_\theta(s,a)
\]

Ở inference:

- Không epsilon.
- Không guided action.
- Không GT reward.
- Chỉ chọn valid action có Q lớn nhất.

Nói ngắn gọn:

```text
Training:
reward → Bellman target → Q-network

Inference:
Q-network → best valid action
```

---

## 15. Điểm dễ nhầm trong source

DQN không tối ưu trực tiếp:

```text
mAP
AP50
số object crop ra
confidence của box
```

Nó tối ưu Q-value của reward thiết kế. Vì vậy nếu reward có lỗi cân bằng, DQN sẽ tối ưu “khe hở” của reward.

Ví dụ:

- Nếu không phạt duplicate: Q có thể cao cho crop vùng đã detect.
- Nếu TP reward quá nhỏ: agent có thể thích ROI compact nhưng không tăng detection.
- Nếu compute cost quá lớn: agent có thể không dám `STOP`.
- Nếu hard-region reward quá lớn nhưng crop reward yếu: agent có thể tối ưu coverage GT hình học thay vì cải thiện detector.

Toàn bộ DQN chỉ “thông minh” theo đúng hàm reward mà nó nhận.