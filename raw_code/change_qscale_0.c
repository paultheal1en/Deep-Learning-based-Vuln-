static void change_qscale(MpegEncContext * s, int dquant)

{

    s->qscale += dquant;



    if (s->qscale < 1)

        s->qscale = 1;

    else if (s->qscale > 31)

        s->qscale = 31;



    s->y_dc_scale= s->y_dc_scale_table[ s->qscale ];

    s->c_dc_scale= s->c_dc_scale_table[ s->qscale ];

}
