for p in ("Knet","JSON","Images","FileIO")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using FileIO

function loaddata()
  info("Loading Visual Genome...")
  impath = "/Users/Aybuke/Desktop/KocUniversity/2017Spring/Comp541/Project/VisualGenome/image_data.json"
  objpath = "/Users/Aybuke/Desktop/KocUniversity/2017Spring/Comp541/Project/VisualGenome/objects.json"
  relpath = "/Users/Aybuke/Desktop/KocUniversity/2017Spring/Comp541/Project/VisualGenome/relationships.json"
  fim = open(impath)
  fobj = open(objpath)
  frel = open(relpath)
  images = JSON.parse(fim)
  objects = JSON.parse(fobj)
  relations = JSON.parse(frel)
  close(fim)
  close(fobj)
  close(frel)
  data = []
  imgInfo = Dict{Any}
  for i in 1:size(images,1)
    path = images[i]["url"]
    img = load(download(path))
    width = images[i]["width"]
    height = images[i]["height"]
    img = convert(Array{Float32},reshape(ImageCore.raw(img),width,height,3))
    imgInfo["image_id"] = images[i]["image_id"]
    imgInfo["image"] = img
    object_bbox = []
    objID = Dict{Any}
    for obj in objects[i]["objects"]
      for j in 1:size(obj,1)
        objID[obj[j]["object_id"]] = j
      end
      x = obj["x"]
      y = obj["y"]
      w = obj["w"]
      h = obj["h"]
      bbox = hcat(x,y,x+w,y+h)
      push!(object_bbox,bbox)
    end
    imgInfo["objects"] = objects_bbox
    relships = []
    for rel in relations[i]["relationships"]
      subj = rel["subject"]["name"]
      obj = rel["object"]["name"]
      pred = rel["predicate"]
      subj_ids = objID[rel["subject"]["object_id"]]
      obj_ids = objID[rel["object"]["object_id"]]
      push!(relships,hcat(subj_ids, obj_ids, subj, pred, obj))
    end
    imgInfo["rels"] = relships
    push!(data,imgInfo)
  end
  partFilePath = "/Users/Aybuke/Desktop/KocUniversity/2017Spring/Comp541/Project/VisualGenome/densecap_splits.json"
  fpart = open(partFilePath)
  sp = JSON.parse(fpart)
  close(fpart)
  dtrn = []
  dtst = []
  dval = []
  for d in data
    if d["image_id"] in sp["test"]
      push!(dtst,d)
    elseif d["image_id"] in sp["train"]
      push!(dtrn,d)
    elseif d["image_id"] in sp["val"]
      push!(dval,d)
    end
  end
	return (dtrn,dtst,dval)
end

# Train 3000000 iteration
# Momentum = 0.95
# Initial LR = 0.005, multiplied with 0.1 after every 120000 iteration
# Each batch = 1 image with all refexp annotated over and image
# Parameters are initilized with Xavier initilizer
global const momentum = 0.95
global lr = 0.005
global embedding_matrix
#####################################
# Expression parsing with attention #
#####################################

# Sequence of T words {wt}
# Embed each word wt to a vector et using GloVe
function parser(w)
  #embedding with GloVe
  ht = lstm(e)
  asubj = attention(?,ht)
  aobj = attention(?,ht)
  arel = attention(?,ht)
  qsubj = languageRep(asubj,e)
  qobj = languageRep(aobj,e)
  qrel = languageRep(arel,e)
  return (qsubj, qobj, qrel)
end
# Scan through the {et} with a 2-layer bidirectonal LSTM
# LTSM = 1000-dim hidden state, ht = 4000-dim
# First layer of LSTM = input: {et}, output: forward hidden state ht(1,fw)
# and backward hidden state ht(1,bw) at each time step. Becomes ht(1)
# Second layer of LSTM = input: {ht(1)}, output: ht(2,fw) & ht(2,bw)
# Then ht = [ht(1,fw), ht(1,bw), ht(2,fw), ht(2,bw)]

function lstmLayer1(e)
  return (ht1fw,ht1bw)
end

function lstmLayer2(ht)
  return (ht2fw, ht2bw)
end
function lstm(e)
  (ht1fw,ht1bw) = lstmLayer1(e)
  ht1 = hcat(ht1fw,ht1bw)
  (ht2fw, ht2bw) = lstmLayer2(ht1)
  ht2 = hcat(ht2fw, ht2bw)
  ht = hcat(ht1,ht2)
  return ht
end
# Attention calculation (subj as ex., rel and obj is same)
# a(t,subj) = exp(B(T,subj).ht) / ∑(µ=1,T) exp(B(T,subj).hµ)
function attention(b,ht)
  expValue = exp(b' * ht)
  a = expValue/sum(expValue)
  return a
end

function languageRep(a,e)
  q=0
  for k = 1:length(e)
    q += attention()*e[k]
  end
  return q
end

#######################
# Localication Module #
#######################

# floc(b,qloc:Qloc)
# Ssubj(bi,bj) = floc(bi,qsubj:Qloc)+floc(bj,qobj:Qloc)+frel(bi,qrel:Qrel)
# Best possible score: Ssubj(bi) = bj-max(Spair(bi,bj))
# Highest scoring region: bsubj*  = bi-argmax(Ssubj(bi))

# Model takes Xvisual and Xspatial
# Xvisual = conv

# Xspatial = [Xmin/WI, Ymin/HI, Xmax/WI, Ymax/HI, Sb/SI],
# where [xmin, ymin, xmax, ymax] and Sb are bounding box coordinates and area of b
# and WI width, HI height and SI are of the image I.
# Spatial Features
function spatial(boundingBox, height, weight)
  box = KnetArray(boundingBox)
  spatialFeatures = zeros(size(box,1),5)
  x1 = box[:, 0] * 2.0 / weight
  y1 = box[:, 1] * 2.0 / height
  x2 = box[:, 2] * 2.0 / weight
  y2 = box[:, 3] * 2.0 / height
  S = (x2-x1) * (y2-y1)

  spatialFeatures[:, 0] = x1
  spatialFeatures[:, 1] = y1
  spatialFeatures[:, 2] = x2
  spatialFeatures[:, 3] = y2
  spatialFeatures[:, 4] = S
  return spatialFeatures
end

# The parameters in QLoc = (Wv,s, bv,s, wloc, bloc)
function locationModule(b,qloc,lw)
  xs = spatial(b,,) # spatial of b
  xv = 0 # visual of b with conv ?!?!?!!?!?!?!?!?!
  # Xv,s = [Xv, Xs] = representation of region b
  xvs = [xv, xs]
  # ~Xv,s = Wv,s*Xv,s + bv,s
  xhat = lw[1]*xvs + lw[2]
  # zloc = ~Xv,s .* qloc
  zloc = xhat .* qloc
  # ^zloc = zloc / ||zloc||
  zhat = zloc / length(zloc)
  # Prediction sloc = wTloc*^zloc+bloc
  sloc = lw[3]' * zhat + lw[4]
  return sloc
end

#######################
# Relationship Module #
#######################

# The parameters in QLoc = (W1,2, b1,2, wrel, brel)
function relationModule(b1,b2,qrel,rw)
  # Xspatial = [Xmin/WI, Ymin/HI, Xmax/WI, Ymax/HI, Sb/SI],
  # where [xmin, ymin, xmax, ymax] and Sb are bounding box coordinates and area of b
  # and WI width, HI height and SI are of the image I.
  x1 = spatial(b1,,) # spatial features of b1
  x2 = spatial(b2,,) # spatial features of b2
  # x1,2 = [x1,x2]
  x = hcat(x1,x2)
  # ~x1,2 = W1,2*x1,2 + b1,2
  xhat = rw[1]*x + rw[2]
  # zrel = ~x1,2 .* Qrel
  zrel = xhat .* qrel
  # ^zrel = zrel / ||zrel||
  zhat = zrel / length(zrel)
  # srel = wTrel * ^zrel + brel
  srel = rw[3]'*zhat + rw[4]
  return srel
end

#######################
# End-to-end Learning #
#######################

# LossStrong= -log(exp(spair(bsubj-gt,bobj-gt))) / ∑ exp(spair(bi,bj)))
function strongLoss()
  subj = locationModule(bsubj,qlocsubj,lwsubj)
  rel = relationModule(bsubj,bobj,qrel,rw)
  obj = locationModule(bobj,qlocobj,lwobj)
  spair = subj+rel+obj
  pairexp = exp(spair)
  loss = -log(pairexp/sum(pairexp))
  return loss
end

# LossWeak = -log(exp(ssubj(bsubj-gt))) / ∑ exp(Ssubj(bi)))
function weakloss()
  subj = locationModule(b,qloc,lw)
  expsubj = exp(subj)
  loss = -log(expsubj/sum(expsubj))
  return loss
end


function precision()
  trueCount = 0
  count = 0
  for (x,y) in data
  end
  result = trueCount / count
  return result
end
