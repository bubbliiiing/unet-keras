from nets.unet import Unet

model = Unet([512,512,3],21)
model.summary()

for i,layer in enumerate(model.layers):
    print(i,layer.name)