from .q_learner import QLearner
from .max_q_learner import MAXQLearner
from .qatten_learner import QattenLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .mude_learner import MUDE_QLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["mude_learner"] = MUDE_QLearner
