static int put_image(struct vf_instance *vf, mp_image_t *mpi, double pts)

{

    mp_image_t *dmpi;

    if (vf->priv->in.fmt == vf->priv->out.fmt) { //nothing to do

        dmpi = mpi;

    } else {

        int out_off_left, out_off_right;

        int in_off_left  = vf->priv->in.row_left   * mpi->stride[0]  +

                           vf->priv->in.off_left;

        int in_off_right = vf->priv->in.row_right  * mpi->stride[0]  +

                           vf->priv->in.off_right;



        dmpi = ff_vf_get_image(vf->next, IMGFMT_RGB24, MP_IMGTYPE_TEMP,

                            MP_IMGFLAG_ACCEPT_STRIDE,

                            vf->priv->out.width, vf->priv->out.height);

        out_off_left   = vf->priv->out.row_left  * dmpi->stride[0] +

                         vf->priv->out.off_left;

        out_off_right  = vf->priv->out.row_right * dmpi->stride[0] +

                         vf->priv->out.off_right;



        switch (vf->priv->out.fmt) {

        case SIDE_BY_SIDE_LR:

        case SIDE_BY_SIDE_RL:

        case SIDE_BY_SIDE_2_LR:

        case SIDE_BY_SIDE_2_RL:

        case ABOVE_BELOW_LR:

        case ABOVE_BELOW_RL:

        case ABOVE_BELOW_2_LR:

        case ABOVE_BELOW_2_RL:

        case INTERLEAVE_ROWS_LR:

        case INTERLEAVE_ROWS_RL:

            memcpy_pic2(dmpi->planes[0] + out_off_left,

                       mpi->planes[0] + in_off_left,

                       3 * vf->priv->width,

                       vf->priv->height,

                       dmpi->stride[0] * vf->priv->row_step,

                       mpi->stride[0] * vf->priv->row_step,

                       vf->priv->row_step != 1);

            memcpy_pic2(dmpi->planes[0] + out_off_right,

                       mpi->planes[0] + in_off_right,

                       3 * vf->priv->width,

                       vf->priv->height,

                       dmpi->stride[0] * vf->priv->row_step,

                       mpi->stride[0] * vf->priv->row_step,

                       vf->priv->row_step != 1);

            break;

        case MONO_L:

        case MONO_R:

            memcpy_pic(dmpi->planes[0],

                       mpi->planes[0] + in_off_left,

                       3 * vf->priv->width,

                       vf->priv->height,

                       dmpi->stride[0],

                       mpi->stride[0]);

            break;

        case ANAGLYPH_RC_GRAY:

        case ANAGLYPH_RC_HALF:

        case ANAGLYPH_RC_COLOR:

        case ANAGLYPH_RC_DUBOIS:

        case ANAGLYPH_GM_GRAY:

        case ANAGLYPH_GM_HALF:

        case ANAGLYPH_GM_COLOR:

        case ANAGLYPH_YB_GRAY:

        case ANAGLYPH_YB_HALF:

        case ANAGLYPH_YB_COLOR: {

            int i,x,y,il,ir,o;

            unsigned char *source     = mpi->planes[0];

            unsigned char *dest       = dmpi->planes[0];

            unsigned int   out_width  = vf->priv->out.width;

            int           *ana_matrix[3];



            for(i = 0; i < 3; i++)

                ana_matrix[i] = vf->priv->ana_matrix[i];



            for (y = 0; y < vf->priv->out.height; y++) {

                o   = dmpi->stride[0] * y;

                il  = in_off_left  + y * mpi->stride[0];

                ir  = in_off_right + y * mpi->stride[0];

                for (x = 0; x < out_width; x++) {

                    dest[o    ]  = ana_convert(

                                   ana_matrix[0], source + il, source + ir); //red out

                    dest[o + 1]  = ana_convert(

                                   ana_matrix[1], source + il, source + ir); //green out

                    dest[o + 2]  = ana_convert(

                                   ana_matrix[2], source + il, source + ir); //blue out

                    il += 3;

                    ir += 3;

                    o  += 3;

                }

            }

            break;

        }

        default:

            ff_mp_msg(MSGT_VFILTER, MSGL_WARN,

                   "[stereo3d] stereo format of output is not supported\n");

            return 0;

            break;

        }

    }

    return ff_vf_next_put_image(vf, dmpi, pts);

}
