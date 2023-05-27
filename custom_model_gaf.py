import tensorflow as tf
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch


class KerasModel(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(KerasModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        
        self.image_15 = tf.keras.layers.Input(shape=original_space['image_15'].shape, name="image_15")
        self.image_d = tf.keras.layers.Input(shape=original_space['image_d'].shape, name="image_d")
        
        self.time = tf.keras.layers.Input(shape=original_space['time'].shape, name="time")
        
       # self.weekday = tf.keras.layers.Input(shape=original_space['weekday'].shape, name="weekday")

        # Concatenating the inputs;
        # One can pass different parts of the state to different networks before concatenation.
        
        conv1_15 = tf.keras.layers.Conv2D(16, (2, 2), (2, 2), activation='tanh', padding='same')(self.image_15)
         
        drop1_15 = tf.keras.layers.Dropout(0.1)(conv1_15)   
        conv2_15 = tf.keras.layers.Conv2D(32, (2, 2), (2, 2), activation='tanh', padding='same')(drop1_15)       
        maxpool1_15 = tf.keras.layers.MaxPool2D((2, 2))(conv2_15)

        drop2_15 = tf.keras.layers.Dropout(0.1)(maxpool1_15)
        conv3_15 = tf.keras.layers.Conv2D(32, (2, 2), (2, 2), activation='tanh', padding='same')(drop2_15)
        #maxpool2_15 = tf.keras.layers.MaxPool2D((2, 2))(conv3_15)
    
        
        conv4_15 = tf.keras.layers.Conv2D(32, (2, 2), (2, 2))(conv3_15)
      
        
        conv_flat_15 = tf.keras.layers.Flatten()(conv4_15)
        conv_out_15 = tf.keras.layers.Dense(1, activation='tanh')(conv_flat_15)
        
        # ----------- DAILY ---------------------
        conv1_d = tf.keras.layers.Conv2D(16, (2, 2), (2, 2), activation='tanh', padding='same')(self.image_d)
               
        drop1_d = tf.keras.layers.Dropout(0.25)(conv1_d)   
        conv2_d = tf.keras.layers.Conv2D(16, (2, 2), (2, 2), activation='tanh', padding='same')(drop1_d)       
        maxpool1_d = tf.keras.layers.MaxPool2D((2, 2))(conv2_d)

        drop2_d = tf.keras.layers.Dropout(0.25)(maxpool1_d)
        conv3_d = tf.keras.layers.Conv2D(32, (2, 2), (2, 2), activation='tanh', padding='same')(drop2_d)
        maxpool2_d = tf.keras.layers.MaxPool2D((2, 2))(conv3_d)
        
      #  conv4_d = tf.keras.layers.Conv2D(32, (2, 2), (2, 2))(maxpool2_d)

        
        conv_flat_d = tf.keras.layers.Flatten()(maxpool2_d)
        conv_out_d = tf.keras.layers.Dense(1, activation='tanh')(conv_flat_d)
        
       # time = tf.keras.layers.Dense((1))(self.time)
        
        concatenated = tf.keras.layers.Concatenate()([conv_out_15, conv_out_d, self.time])

        # Building the dense layers
        layer_out = tf.keras.layers.Dense(num_outputs, activation='tanh')(concatenated)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(concatenated)
        
        self.base_model = tf.keras.Model([self.image_15, self.image_d, self.time], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        inputs = {'image_15': orig_obs["image_15"], 'image_d': orig_obs["image_d"], 'time': orig_obs['time']}
        model_out, self._value_out = self.base_model(inputs) 
        
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])  
    

class Multi_Vol(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(Multi_Vol, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        
        self.image = tf.keras.layers.Input(shape=original_space['image'].shape, name="image")

        self.vol = tf.keras.layers.Input(shape=original_space['vol'].shape, name="vol")
        
        # IMAGE CONV
        conv1 = tf.keras.layers.Conv1D(8, 2, activation='tanh', padding='valid')(self.image)
        drop1 = tf.keras.layers.Dropout(0.1)(conv1)   
        conv2 = tf.keras.layers.Conv1D(32, 2, activation='tanh', padding='valid')(drop1)       
        maxpool1 = tf.keras.layers.MaxPool1D(2, 2)(conv2)
        drop2 = tf.keras.layers.Dropout(0.1)(maxpool1)
        conv3 = tf.keras.layers.Conv1D(32, 2, activation='tanh', padding='same')(drop2)
        maxpool2 = tf.keras.layers.MaxPool1D(2, 2)(conv3)
        conv4 = tf.keras.layers.Conv1D(32, 2, activation='tanh')(maxpool2)
        
        conv_flat = tf.keras.layers.Flatten()(conv4)
        conv_out = tf.keras.layers.Dense(1, activation='tanh')(conv_flat)
             
        
        concatenated = tf.keras.layers.Concatenate()([conv_out, self.vol])
        layer_out = tf.keras.layers.Dense(num_outputs, activation='tanh')(concatenated)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(concatenated)
        
        self.base_model = tf.keras.Model([self.image, self.vol], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        inputs = {'image': orig_obs["image"], 'vol': orig_obs['vol']}
        model_out, self._value_out = self.base_model(inputs) 
        
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])  
    
class Continuous_CNN(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(Continuous_CNN, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        
        self.image = tf.keras.layers.Input(shape=original_space['image'].shape, name="image")
        self.position = tf.keras.layers.Input(shape=original_space['position'].shape, name='position')
        self.vol = tf.keras.layers.Input(shape=original_space['vol'].shape, name="vol")
        
        # IMAGE CONV
        conv1 = tf.keras.layers.Conv1D(8, 2, activation='tanh', padding='valid')(self.image)
        maxpool1 = tf.keras.layers.MaxPool1D(2, 2)(conv1)
        drop1 = tf.keras.layers.Dropout(0.1)(maxpool1)   
        conv2 = tf.keras.layers.Conv1D(32, 2, activation='tanh', padding='valid')(drop1)       
        maxpool2 = tf.keras.layers.MaxPool1D(2, 2)(conv2)
        drop2 = tf.keras.layers.Dropout(0.1)(maxpool2)
        conv3 = tf.keras.layers.Conv1D(32, 2, activation='tanh', padding='same')(drop2)
        maxpool3 = tf.keras.layers.MaxPool1D(2, 2)(conv3)
        drop3 = tf.keras.layers.Dropout(0.1)(maxpool3) 
        conv4 = tf.keras.layers.Conv1D(32, 2, activation='tanh')(drop3)
        
        conv_flat = tf.keras.layers.Flatten()(conv4)
        conv_out = tf.keras.layers.Dense(1, activation='tanh')(conv_flat)
             
        
        concatenated = tf.keras.layers.Concatenate()([conv_out, self.position, self.vol])
        layer_out = tf.keras.layers.Dense(num_outputs, activation='tanh')(concatenated)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(concatenated)
        
        self.base_model = tf.keras.Model([self.image, self.position,self.vol], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        inputs = {'image': orig_obs["image"], 'position': orig_obs['position'],'vol': orig_obs['vol']}
        self.base_model.summary()
        model_out, self._value_out = self.base_model(inputs) 
        
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])  

class Continuous_CNN_15_4h(TFModelV2):
    """Custom model for PPO."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(Continuous_CNN_15_4h, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        
        self.image = tf.keras.layers.Input(shape=original_space['image'].shape, name="image")
        self.position = tf.keras.layers.Input(shape=original_space['position'].shape, name='position')
        self.vol = tf.keras.layers.Input(shape=original_space['vol'].shape, name="vol")
        
        # IMAGE CONV
        conv1 = tf.keras.layers.Conv2D(32, (2, 2), groups=2, activation='tanh', padding='valid')(self.image)
        maxpool1 = tf.keras.layers.MaxPool2D(3, 3)(conv1)
        drop1 = tf.keras.layers.Dropout(0.1)(maxpool1)   
        conv2 = tf.keras.layers.Conv2D(48, (2, 2), groups=2, activation='tanh', padding='valid')(drop1)       
        maxpool2 = tf.keras.layers.MaxPool2D(3, 3)(conv2)
        drop2 = tf.keras.layers.Dropout(0.1)(maxpool2)
        conv3 = tf.keras.layers.Conv2D(64, (2, 2), groups=2, activation='tanh', padding='same')(drop2)
        maxpool3 = tf.keras.layers.MaxPool2D(3, 3)(conv3)
        drop3 = tf.keras.layers.Dropout(0.1)(maxpool3) 
        conv4 = tf.keras.layers.Conv2D(64, (2, 2), groups=2, activation='tanh')(drop3)
        maxpool4 = tf.keras.layers.MaxPool2D(2, 2)(conv4)
        
        conv_flat = tf.keras.layers.Flatten()(maxpool4)
        conv_out = tf.keras.layers.Dense(1, activation='tanh')(conv_flat)
             
        
        concatenated = tf.keras.layers.Concatenate()([conv_out, self.position, self.vol])
        layer_out = tf.keras.layers.Dense(num_outputs, activation='tanh')(concatenated)
        
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(concatenated)
        
        self.base_model = tf.keras.Model([self.image, self.position,self.vol], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        inputs = {'image': orig_obs["image"], 'position': orig_obs['position'],'vol': orig_obs['vol']}
        model_out, self._value_out = self.base_model(inputs) 
        
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])  