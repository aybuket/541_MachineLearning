using Knet
include("TermProject.jl")
# Visual Genome weak full model
function main()
  dtrn, dtst, dval = loaddata()
end

lossgradient = grad(weakLoss)
function train(data)
  for epoch=1:300000
    for d in data
      g = lossgradient()
      m = Momentum(lr=learning_rate, gamma=momentum)
      update!(w,g,m)
    end
    if epoch % 160000 == 0
      learning_rate = learning_rate*0.1
    end
  end
end

main()
