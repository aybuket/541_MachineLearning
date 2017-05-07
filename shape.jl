using Knet
using Pycall
@pyimport numpy
include("TermProject.jl")
include("vgg.jl")
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
imcrop_batch = Array{Float32}(N_bbox, IM_H, IM_W, 3)
spatial_batch = Array{Float32}(N_bbox, 5)
text_seq_batch = Array{Int32}(T, 1)
label_batch = Array{Int32}(1)

function main()
  trn = np.load('trn.npz')
  tst = np.load('tst.npz')
  val = np.load('val.npz')
  # returns a dictionary which has following entries:
  # parsed_query_list
  # query_list
  # meta_list
  # matched_pairs_list
  # image_list
  vocab_file = open("vocabular_72700.txt")
  # for every image use VGG.main(imagefile) to extract visual features

end

lossgradient = grad(weakLoss)
function train(data,w)
  for epoch=1:300000
    for d in data
      g = lossgradient() # inputs are b,qloc,lw
      m = Momentum(lr=learning_rate, gamma=momentum)
      update!(w,g,m)
    end
    if epoch % 160000 == 0
      learning_rate = learning_rate*0.1
    end
  end
end

main()
