import numpy as np

import matplotlib.pyplot as plt

# Extracting and organizing data from the file content
# Accuracy values for 70 rounds and 1 epoch per round
accuracy_1_epoch = [
    0.0989, 0.103, 0.2345, 0.2634, 0.2803, 0.2976, 0.3152, 0.3448, 0.3329, 
    0.3507, 0.3494, 0.3507, 0.3606, 0.3615, 0.3712, 0.3565, 0.3775, 0.3799, 
    0.3777, 0.3776, 0.382, 0.3834, 0.3915, 0.4016, 0.3993, 0.4049, 0.4146, 
    0.425, 0.4097, 0.4274, 0.4204, 0.434, 0.4262, 0.4197, 0.4338, 0.4276, 
    0.4347, 0.436, 0.4408, 0.436, 0.4459, 0.4497, 0.4417, 0.4478, 0.4375, 
    0.4252, 0.4371, 0.4466, 0.4412, 0.4487, 0.438, 0.4572, 0.4556, 0.4413, 
    0.4512, 0.4324, 0.4554, 0.4424, 0.4472, 0.4494, 0.4368, 0.4559, 0.4369, 
    0.4273, 0.4379, 0.4525, 0.4518, 0.44, 0.4304, 0.4223
]

# Accuracy values for 70 rounds and 2 epochs per round
accuracy_2_epochs = [
    0.1004, 0.205, 0.2741, 0.2972, 0.3434, 0.3475, 0.3745, 0.3732, 0.3865, 
    0.4062, 0.4196, 0.4284, 0.4327, 0.4332, 0.4359, 0.4425, 0.4374, 0.4461, 
    0.4553, 0.4529, 0.4642, 0.4571, 0.4566, 0.4549, 0.459, 0.4643, 0.4677, 
    0.4607, 0.4677, 0.4726, 0.4618, 0.477, 0.447, 0.4588, 0.4705, 0.478, 
    0.4778, 0.4619, 0.4544, 0.4732, 0.4721, 0.4763, 0.4635, 0.4783, 0.4618, 
    0.4528, 0.4741, 0.4569, 0.4618, 0.4797, 0.4543, 0.4689, 0.4439, 0.4025, 
    0.4723, 0.4771, 0.4515, 0.4606, 0.4659, 0.4651, 0.4694, 0.4541, 0.4676, 
    0.4432, 0.4732, 0.4429, 0.4656, 0.4651, 0.4472, 0.4594
]

# Accuracy values for 70 rounds and 3 epochs per round
accuracy_3_epochs = [
    0.103, 0.1894, 0.323, 0.3548, 0.3661, 0.3908, 0.4007, 0.4073, 0.4202, 
    0.4262, 0.4439, 0.4513, 0.4563, 0.459, 0.4634, 0.4649, 0.4672, 0.4748, 
    0.4583, 0.4748, 0.4695, 0.4668, 0.477, 0.4655, 0.4794, 0.474, 0.4826, 
    0.4829, 0.4294, 0.4704, 0.4718, 0.4723, 0.4738, 0.4565, 0.4561, 0.4579, 
    0.4775, 0.468, 0.4516, 0.4596, 0.4434, 0.4461, 0.4726, 0.4679, 0.4452, 
    0.4555, 0.4584, 0.4486, 0.4784, 0.4455, 0.4145, 0.4196, 0.4252, 0.4584, 
    0.4231, 0.4235, 0.4021, 0.3991, 0.4214, 0.4701, 0.3938, 0.4236, 0.448, 
    0.4803, 0.4238, 0.4412, 0.3752, 0.4626, 0.438, 0.4237
]

accuracy_4_epochs = [
    0.0978, 0.2442, 0.337, 0.3816, 0.4132, 0.4191, 0.4262, 0.4335, 0.4486, 
    0.4553, 0.4639, 0.4689, 0.4703, 0.4654, 0.4645, 0.482, 0.4617, 0.4686, 
    0.4898, 0.4637, 0.4759, 0.478, 0.4873, 0.4899, 0.489, 0.4883, 0.4823, 
    0.4596, 0.4873, 0.4644, 0.4605, 0.4819, 0.4913, 0.4743, 0.501, 0.4783, 
    0.4688, 0.4794, 0.474, 0.4463, 0.391, 0.3983, 0.394, 0.4376, 0.3968, 
    0.3341, 0.4629, 0.4907, 0.4435, 0.3736, 0.4466, 0.4318, 0.4089, 0.423, 
    0.4115, 0.3934, 0.3768, 0.4342, 0.4066, 0.4046, 0.3424, 0.4, 0.3494, 
    0.3614, 0.3723, 0.299, 0.3174, 0.3095, 0.355, 0.4195
]

# Accuracy values for 70 rounds and 5 epochs per round
accuracy_5_epochs = [
    0.0868, 0.2275, 0.3214, 0.3692, 0.3909, 0.4065, 0.423, 0.4222, 0.4373, 
    0.4373, 0.4421, 0.4436, 0.4484, 0.4504, 0.4592, 0.3146, 0.4451, 0.3799, 
    0.4581, 0.4518, 0.4609, 0.4509, 0.4323, 0.3485, 0.4538, 0.4466, 0.45, 
    0.4518, 0.4442, 0.4603, 0.3723, 0.4259, 0.4194, 0.4426, 0.4296, 0.4559, 
    0.4298, 0.3922, 0.4536, 0.4363, 0.4184, 0.4535, 0.403, 0.3915, 0.4131, 
    0.4647, 0.4077, 0.4467, 0.4186, 0.3767, 0.4509, 0.4656, 0.4002, 0.4333, 
    0.3626, 0.359, 0.3251, 0.3874, 0.3761, 0.4309, 0.4255, 0.2913, 0.3546, 
    0.2954, 0.384, 0.4361, 0.3734, 0.3724, 0.4468, 0.3806
]

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Define the window size for the moving average
window_size = 10
# Calculate moving averages
ma_accuracy_1_epoch = moving_average(accuracy_1_epoch, window_size)
ma_accuracy_2_epochs = moving_average(accuracy_2_epochs, window_size)
ma_accuracy_3_epochs = moving_average(accuracy_3_epochs, window_size)
ma_accuracy_4_epochs = moving_average(accuracy_4_epochs, window_size)
ma_accuracy_5_epochs = moving_average(accuracy_5_epochs, window_size)

# Plotting the moving average accuracy values
plt.figure(figsize=(10, 6))
plt.plot(ma_accuracy_1_epoch, label='1 Epoch per Round')
plt.plot(ma_accuracy_2_epochs, label='2 Epochs per Round')
plt.plot(ma_accuracy_3_epochs, label='3 Epochs per Round')
plt.plot(ma_accuracy_4_epochs, label='4 Epochs per Round')
plt.plot(ma_accuracy_5_epochs, label='5 Epochs per Round')
plt.title(f'Moving Average Accuracy for Different Epochs per Round (Window Size = {window_size})')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./figures/moving_average_accuracy.png', dpi=500)
# plt.show()

# Plotting the raw accuracy values
plt.figure(figsize=(10, 6))
plt.plot(accuracy_1_epoch, label='1 Epoch per Round')
plt.plot(accuracy_2_epochs, label='2 Epochs per Round')
plt.plot(accuracy_3_epochs, label='3 Epochs per Round')
plt.plot(accuracy_4_epochs, label='4 Epochs per Round')
plt.plot(accuracy_5_epochs, label='5 Epochs per Round')
plt.title('Raw Accuracy over 70 Rounds for Different Epochs per Round')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./figures/raw_accuracy.png', dpi=500)
# plt.show()

