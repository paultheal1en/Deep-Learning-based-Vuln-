static void decode_cdlms(WmallDecodeCtx *s)

{

    int c, i;

    int cdlms_send_coef = get_bits1(&s->gb);



    for(c = 0; c < s->num_channels; c++) {

	s->cdlms_ttl[c] = get_bits(&s->gb, 3) + 1;

	for(i = 0; i < s->cdlms_ttl[c]; i++) {

	    s->cdlms[c][i].order = (get_bits(&s->gb, 7) + 1) * 8;

	}



	for(i = 0; i < s->cdlms_ttl[c]; i++) {

	    s->cdlms[c][i].scaling = get_bits(&s->gb, 4);

	}



	if(cdlms_send_coef) {

	    for(i = 0; i < s->cdlms_ttl[c]; i++) {

		int cbits, shift_l, shift_r, j;

		cbits = av_log2(s->cdlms[c][i].order);

		if(1 << cbits < s->cdlms[c][i].order)

		    cbits++;

		s->cdlms[c][i].coefsend = get_bits(&s->gb, cbits) + 1;



		cbits = av_log2(s->cdlms[c][i].scaling + 1);

		if(1 << cbits < s->cdlms[c][i].scaling + 1)

		    cbits++;



		s->cdlms[c][i].bitsend = get_bits(&s->gb, cbits) + 2;

		shift_l = 32 - s->cdlms[c][i].bitsend;

		shift_r = 32 - 2 - s->cdlms[c][i].scaling;

		for(j = 0; j < s->cdlms[c][i].coefsend; j++) {

		    s->cdlms[c][i].coefs[j] =

			(get_bits(&s->gb, s->cdlms[c][i].bitsend) << shift_l) >> shift_r;

		}

	    }

	}

    }

}
