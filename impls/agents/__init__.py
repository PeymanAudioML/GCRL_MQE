from agents.cmd import CMDAgent
from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.mqe import MQEAgent
from agents.ngcsacbc import NGCSACBCAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.tmd import TMDAgent

agents = dict(
    cmd=CMDAgent,
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    mqe=MQEAgent,
    ngcsacbc=NGCSACBCAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    tmd=TMDAgent,
)
