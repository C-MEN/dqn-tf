import os
import pprint
import inspect
import tensorflow as tf

pp = pprint.PrettyPrinter().pprint

def class_vars(obj):
  return {k:v for k, v in inspect.getmembers(obj)
      if not k.startswith('__') and not callable(k)}

class BaseModel(object):
  """Abstract object representing an Reader model."""
  def __init__(self, config):
    self._saver = None
    self.config = config

    try:
      self._attrs = config.__dict__['__flags']
    except:
      self._attrs = class_vars(config)
    pp(self._attrs)

    self.config = config

    for attr in self._attrs:
      name = attr if not attr.startswith('_') else attr[1:]
      setattr(self, name, getattr(self.config, attr))

  def save_model(self, step=None):
    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__
    model_path = os.path.join(self.checkpoint_dir, "model")
    
    if not os.path.exists(self.checkpoint_dir):
      os.makedirs(self.checkpoint_dir)
    self.saver.save(self.sess, model_path, global_step=step)

  def load_model(self):
    print(" [*] Loading checkpoints...")

    chkpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if chkpt and chkpt.model_checkpoint_path:
      self.saver.restore(self.sess, chkpt.model_checkpoint_path)
      print(" [*] Load SUCCESS: %s" % chkpt.model_checkpoint_path)
      return True
    else:
      print(" [!] Load FAILED: %s" % self.checkpoint_dir)
      return False

  @property
  def checkpoint_dir(self):
    return os.path.join('checkpoints', self.model_dir)

  @property
  def model_dir(self):
    path_parts = [self.config.env_name]
    for k, v in self._attrs.items():
      if not k.startswith('_') and k not in ['display']:
        path_parts.append("%s-%s" % (k, ",".join([str(i) for i in v]) if type(v) == list else v))
    path_parts.sort()
    return os.path.sep.join(path_parts)

  @property
  def saver(self):
    if self._saver == None:
      self._saver = tf.train.Saver(max_to_keep=10)
    return self._saver
