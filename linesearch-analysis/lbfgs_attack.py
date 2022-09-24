"""
This module provide the attack method of "LBFGS".
"""
from __future__ import division

from builtins import range
import logging

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .base import Attack

__all__ = ['LBFGSAttack', 'LBFGS']

"""
The base model of the model.
"""
from builtins import object
import logging
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from future.utils import with_metaclass


class Attack(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for adversarial attacks. `Attack` represent an
    adversarial attack which search an adversarial example. subclass should
    implement the _apply() method.

    Args:
        model(Model): an instance of the class adversarialbox.base.Model.

    """

    def __init__(self, model):
        self.model = model

    def __call__(self, adversary, **kwargs):
        """
        Generate the adversarial sample.

        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.
        """
        self._preprocess(adversary)
        return self._apply(adversary, **kwargs)

    @abstractmethod
    def _apply(self, adversary, **kwargs):
        """
        Search an adversarial example.

        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.
        """
        raise NotImplementedError

    def _preprocess(self, adversary):
        """
        Preprocess the adversary object.

        :param adversary: adversary
        :return: None
        """
        # assert self.model.channel_axis() == adversary.original.ndim

        if adversary.original_label is None:
            adversary.original_label = np.argmax(
                self.model.predict(adversary.original))
        if adversary.is_targeted_attack and adversary.target_label is None:
            if adversary.target is None:
                raise ValueError(
                    'When adversary.is_targeted_attack is true, '
                    'adversary.target_label or adversary.target must be set.')
            else:
                adversary.target_label = np.argmax(
                    self.model.predict(adversary.target))

        logging.info('adversary:'
                     '\n         original_label: {}'
                     '\n         target_label: {}'
                     '\n         is_targeted_attack: {}'
                     ''.format(adversary.original_label, adversary.target_label,
                               adversary.is_targeted_attack))


class LBFGSAttack(Attack):
    """
    Uses L-BFGS-B to minimize the cross-entropy and the distance between the
    original and the adversary.

    Paper link: https://arxiv.org/abs/1510.05328
    """

    def __init__(self, model):
        super(LBFGSAttack, self).__init__(model)
        self._predicts_normalized = None
        self._adversary = None  # type: Adversary

    def _apply(self, adversary, epsilon=0.001, steps=10):
        # 如何查找最佳的C呢
        # step 1, 首先使用指数增长的方式找到一个合适的C1可以攻击成功, 假设C0会让攻击失败, C1 = 2*C0, C1会让攻击成功.
        # step 2, 然后再[0,C1]使用二分查找的方式找到一个合适的且可以攻击成功的C2, 其实这里可以使用 step 1里的失败值C0作为初始值来进行区间的二分搜索.[C0,C1].
        self._adversary = adversary

        if not adversary.is_targeted_attack:
            raise ValueError("This attack method only support targeted attack!")

        # finding initial c
        logging.info('finding initial c...')
        c = epsilon
        x0 = np.copy(adversary.original.flatten())
        for i in range(30):
            c = 2 * c
            logging.info('c={}'.format(c))
            is_adversary = self._lbfgsb(x0, c, steps)
            if is_adversary:
                break
        if not is_adversary:
            logging.info('Failed!')
            return adversary

        # binary search c
        logging.info('binary search c...')
        c_low = 0
        c_high = c
        while c_high - c_low >= epsilon:
            logging.info('c_high={}, c_low={}, diff={}, epsilon={}'
                         .format(c_high, c_low, c_high - c_low, epsilon))
            c_half = (c_low + c_high) / 2
            is_adversary = self._lbfgsb(x0, c_half, steps)
            if is_adversary:
                c_high = c_half
            else:
                c_low = c_half

        return adversary

    def _is_predicts_normalized(self, predicts):
        """
        To determine the predicts is normalized.
        :param predicts(np.array): the output of the model.
        :return: bool
        """
        if self._predicts_normalized is None:
            if self.model.predict_name().lower() in [
                'softmax', 'probabilities', 'probs'
            ]:
                self._predicts_normalized = True
            else:
                if np.any(predicts < 0.0):
                    self._predicts_normalized = False
                else:
                    s = np.sum(predicts.flatten())
                    if 0.999 <= s <= 1.001:
                        self._predicts_normalized = True
                    else:
                        self._predicts_normalized = False
        assert self._predicts_normalized is not None
        return self._predicts_normalized

    def _loss(self, adv_x, c):
        """
        To get the loss and gradient.
        :param adv_x: the candidate adversarial example
        :param c: parameter 'C' in the paper
        :return: (loss, gradient)
        """
        x = adv_x.reshape(self._adversary.original.shape)

        # cross_entropy
        # 这里就是得到模型的概率值与目标label之间的交叉熵, 目标label不是正确的label
        logits = self.model.predict(x)
        if not self._is_predicts_normalized(logits):  # to softmax
            e = np.exp(logits)
            logits = e / np.sum(e)
        e = np.exp(logits)
        s = np.sum(e)
        ce = np.log(s) - logits[self._adversary.target_label]

        # L2 distance
        min_, max_ = self.model.bounds()
        d = np.sum((x - self._adversary.original).flatten() ** 2) \
            / ((max_ - min_) ** 2) / len(adv_x)

        # gradient
        gradient = self.model.gradient(x, self._adversary.target_label)
        # 既要最小化预测值与target label之间的 cross entropy loss 又要最小化范数 loss
        result = (c * ce + d).astype(float), gradient.flatten().astype(float)
        return result

    def _lbfgsb(self, x0, c, maxiter):
        # bounds 相当于对变量每一维度的最大值和最小值做一个约束,假设变量有n维,
        # 那 bounds由长度为n的列表组成, 列表中的每一个元素为一个元组,元组包含这一维度能达到的最大值与最小值
        min_, max_ = self.model.bounds()
        bounds = [(min_, max_)] * len(x0)
        approx_grad_eps = (max_ - min_) / 100.0
        x, f, d = fmin_l_bfgs_b(
            self._loss,
            x0,
            args=(c,),
            bounds=bounds,
            maxiter=maxiter,
            epsilon=approx_grad_eps)
        if np.amax(x) > max_ or np.amin(x) < min_:
            x = np.clip(x, min_, max_)
        shape = self._adversary.original.shape
        adv_label = np.argmax(self.model.predict(x.reshape(shape)))
        logging.info('pre_label = {}, adv_label={}'.format(
            self._adversary.target_label, adv_label))
        return self._adversary.try_accept_the_example(
            x.reshape(shape), adv_label)


LBFGS = LBFGSAttack
