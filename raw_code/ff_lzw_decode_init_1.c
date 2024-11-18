int ff_lzw_decode_init(LZWState *p, int csize, uint8_t *buf, int buf_size, int mode)

{

    struct LZWState *s = (struct LZWState *)p;



    if(csize < 1 || csize > LZW_MAXBITS)

        return -1;

    /* read buffer */

    s->eob_reached = 0;

    s->pbuf = buf;

    s->ebuf = s->pbuf + buf_size;

    s->bbuf = 0;

    s->bbits = 0;

    s->bs = 0;



    /* decoder */

    s->codesize = csize;

    s->cursize = s->codesize + 1;

    s->curmask = mask[s->cursize];

    s->top_slot = 1 << s->cursize;

    s->clear_code = 1 << s->codesize;

    s->end_code = s->clear_code + 1;

    s->slot = s->newcodes = s->clear_code + 2;

    s->oc = s->fc = 0;

    s->sp = s->stack;



    s->mode = mode;

    switch(s->mode){

    case FF_LZW_GIF:

        s->extra_slot= 0;

        break;

    case FF_LZW_TIFF:

        s->extra_slot= 1;

        break;

    default:

        return -1;

    }

    return 0;

}
