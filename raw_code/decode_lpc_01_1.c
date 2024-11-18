decode_lpc(WmallDecodeCtx *s)

{

    int ch, i, cbits;

    s->lpc_order = get_bits(&s->gb, 5) + 1;

    s->lpc_scaling = get_bits(&s->gb, 4);

    s->lpc_intbits = get_bits(&s->gb, 3) + 1;

    cbits = s->lpc_scaling + s->lpc_intbits;

    for(ch = 0; ch < s->num_channels; ch++) {

	for(i = 0; i < s->lpc_order; i++) {

	    s->lpc_coefs[ch][i] = get_sbits(&s->gb, cbits);

	}

    }

}
