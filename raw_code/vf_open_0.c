static int vf_open(vf_instance_t *vf, char *args){

    vf->config=config;

    vf->put_image=put_image;

    vf->get_image=get_image;

    vf->query_format=query_format;

    vf->uninit=uninit;

    vf->control= control;

    vf->priv=malloc(sizeof(struct vf_priv_s));

    memset(vf->priv, 0, sizeof(struct vf_priv_s));



    if (args) sscanf(args, "%d:%d", &vf->priv->qp, &vf->priv->mode);



    if(vf->priv->qp < 0)

        vf->priv->qp = 0;



    init_thres2();



    switch(vf->priv->mode){

        case 0: requantize= hardthresh_c; break;

        case 1: requantize= softthresh_c; break;

        default:

        case 2: requantize= mediumthresh_c; break;

    }



#if HAVE_MMX

    if(ff_gCpuCaps.hasMMX){

        dctB= dctB_mmx;

    }

#endif

#if 0

    if(ff_gCpuCaps.hasMMX){

        switch(vf->priv->mode){

            case 0: requantize= hardthresh_mmx; break;

            case 1: requantize= softthresh_mmx; break;

        }

    }

#endif



    return 1;

}
