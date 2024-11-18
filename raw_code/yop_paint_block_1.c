static void yop_paint_block(YopDecContext *s, int tag)

{

    s->dstptr[0]                        = s->srcptr[0];

    s->dstptr[1]                        = s->srcptr[paint_lut[tag][0]];

    s->dstptr[s->frame.linesize[0]]     = s->srcptr[paint_lut[tag][1]];

    s->dstptr[s->frame.linesize[0] + 1] = s->srcptr[paint_lut[tag][2]];



    // The number of src bytes consumed is in the last part of the lut entry.

    s->srcptr += paint_lut[tag][3];

}
