import tensorflow as tf

from niftynet.application.segmentation_application import \
    SegmentationApplication
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE, NETWORK_OUTPUT
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.layer.loss_segmentation import LossFunction

import numpy as np

SUPPORTED_INPUT = set(['image', 'label', 'weight'])

def is_tuple(var):
    return isinstance(var, tuple)

def is_list(var):
    return isinstance(var, list)

def annealing_cos(start, end, pct:float):
    cos_out = np.cos(np.pi*pct)+1
    return end + (start-end)/2 * cos_out

class Scheduler():
    def __init__(self, vals, n_iter:int):
        self.start, self.end = (vals[0], vals[1]) if is_tuple(vals) else (vals, 0)
        self.n_iter = max(1, n_iter)
        self.func = annealing_cos
        self.n = 0

    def restart(self):
        self.n = 0

    def step(self):
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)

    @property
    def is_done(self)->bool:
        return self.n >= self.n_iter

def steps(dico):
    return [Scheduler(step, n_iter) for (step, n_iter) in zip(dico['steps_cfg'], dico['phases'])]

def rule(train, lr_scheds, mom_scheds, idx_s):
    if train :
        if idx_s >= len(lr_scheds):
            return {'stop' : True}
        lr = lr_scheds[idx_s].step()
        mom = mom_scheds[idx_s].step()
        if lr_scheds[idx_s].is_done:
            idx_s += 1
        return {'lr' : lr, 'mom' : mom,'idx' : idx_s}

class DecayLearningRateApplication(SegmentationApplication):
    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, is_training):
        SegmentationApplication.__init__(
            self, net_param, action_param, is_training)
        tf.logging.info('starting decay learning segmentation application')
        self.learning_rate = None
        self.momentum = None
        max_lr = action_param.lr
        self.max = action_param.max_iter
        pct = 1/max_lr if max_lr > 3 else 0.3
        a = int(self.max*pct)
        b = self.max-a
        phases = (a,b)

        div_factor = 20
        final_div = div_factor * 1e3
        low_lr = max_lr/div_factor
        min_lr = max_lr/final_div
        lr_cfg = ((low_lr, max_lr), (max_lr, min_lr))
        moms = (action_param.mom, action_param.mom_end)
        mom_cfg=(moms,(moms[1],moms[0]))

        self.lr_prop = steps({'steps_cfg':lr_cfg, 'phases':phases})
        self.mom_prop = steps({'steps_cfg':mom_cfg,'phases':phases})
        self.current_lr = self.lr_prop[0].start
        self.mom = self.mom_prop[0].start
        self.res = {}
        # print("\n\nThe maximum learning rate should be greater than 1e-3\n\n")

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        if self.is_training:
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(True),
                                    lambda: switch_sampler(False))
            else:
                data_dict = switch_sampler(True)

            image = tf.cast(data_dict['image'], tf.float32)
            net_out = self.net(image, self.is_training)

            with tf.name_scope('Optimiser'):
                self.learning_rate = tf.placeholder(tf.float64, shape=[])
                self.momentum = tf.placeholder(tf.float64, shape=[])
                assert self.action_param.optimiser == 'momentum'
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                   learning_rate=self.learning_rate, momentum=self.momentum)
            loss_func = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type)
            data_loss = loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None),
                weight_map=data_dict.get('weight', None))

            self.current_loss = data_loss
            loss = data_loss
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = data_loss + reg_loss
            grads = self.optimiser.compute_gradients(loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=self.current_loss, name='dice_loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.learning_rate, name='lr',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.momentum, name='mom',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='dice_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            SegmentationApplication.connect_data_and_network(
                self, outputs_collector, gradients_collector)

    def set_iteration_update(self, iteration_message):
        """
        This function will be called by the application engine at each
        iteration.
        """
        current_iter = iteration_message.current_iter
        if iteration_message.is_training:
            self.res = rule(iteration_message.is_training, self.lr_prop, self.res.get('idx', 0))
            iteration_message.should_stop = self.res.get('stop', False)
            self.current_lr = self.res.get('lr',0)
            self.mom = self.res.get('mom',1)

            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            iteration_message.data_feed_dict[self.is_validation] = True
        iteration_message.data_feed_dict[self.learning_rate] = self.current_lr
        iteration_message.data_feed_dict[self.momentum] = self.mom
