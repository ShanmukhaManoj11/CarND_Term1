import pickle
import matplotlib.pyplot as plt

with open('./losses_NVIDIA.p','rb') as f:
    losses=pickle.load(f)
with open('./val_losses_NVIDIA.p','rb') as f:
    val_losses=pickle.load(f)
x=[0,1,2,3,4,5,6,7,8,9]

plt.plot(x,losses,'b.-',label='loss')
plt.plot(x,val_losses,'g.-',label='val_loss')
plt.legend()
plt.show()

plt.plot(x,val_losses,'g.-',label='val_loss')
plt.legend()
plt.show()

