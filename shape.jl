using Knet
using Pycall
@pyimport numpy
include("TermProject.jl")
global const np = PyCall.pywrap(PyCall.pyimport("numpy"))
# Shape

# Shape Parameters
T = 20
N_bbox = 25
IM_H = 224
IM_W = 224
# learning_rate will descent 0.1 every 10.000 step
# max iteration 25.000
weight_decay = 0.0005
imcrop_batch = KnetArray([N_bbox, IM_H, IM_W, 3])
spatial_batch = tf.placeholder(tf.float32, [N_bbox, 5])
text_seq_batch = tf.placeholder(tf.int32, [T, 1])
label_batch = tf.placeholder(tf.int32, [1])

function main()
  trn = np.load('trn.npz')
  tst = np.load('tst.npz')
  val = np.load('val.npz')
  vocab_file = np.load("")

end
lossgradient = grad(weakLoss)
function train(data)
  for epoch=1:300000
    for d in data
      g = lossgradient()
      update!(w,g,;lr=learning_rate)
    end
    if epoch % 160000 == 0
      learning_rate = learning_rate*0.1
    end
  end
end

main()
