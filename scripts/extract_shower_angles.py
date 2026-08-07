import ROOT as RT
from utils import read_header, provide_list
import sys
import numpy as np

from argparse import ArgumentParser as ap

if __name__ == '__main__':

    parser = ap()
    parser.add_argument('-i', type=str, help='Input', required=True)
    parser.add_argument('-o', type=str, help='Output', required=True)
    parser.add_argument('--tag', type=str, help='Larsoft tag', required=True)
    args = parser.parse_args()
    read_header('gallery/ValidHandle.h')
    mcpartv = 'std::vector<simb::MCTruth>'
    classes = [mcpartv]
    provide_list(classes)

    ev = RT.gallery.Event(RT.vector(RT.string)(1, args.i))
    get_parts = ev.getValidHandle[mcpartv]
    vals = []
    for i in range(ev.numberOfEventsInFile()):
        ev.goToEntry(i)
        truths = get_parts(RT.art.InputTag(args.tag))

        truth = truths.product()[0]

        part = truth.GetParticle(0)
        vals.append([part.Px(), part.Py(), part.Pz()])
        # print(vals)

    np.save(args.o, np.array(vals))