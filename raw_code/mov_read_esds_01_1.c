static int mov_read_esds(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    return ff_mov_read_esds(c->fc, pb, atom);

}
