static void set_frame_data(MIContext *mi_ctx, int alpha, AVFrame *avf_out)

{

    int x, y, plane;



    for (plane = 0; plane < mi_ctx->nb_planes; plane++) {

        int width = avf_out->width;

        int height = avf_out->height;

        int chroma = plane == 1 || plane == 2;



        for (y = 0; y < height; y++)

            for (x = 0; x < width; x++) {

                int x_mv, y_mv;

                int weight_sum = 0;

                int i, val = 0;

                Pixel *pixel = &mi_ctx->pixels[x + y * avf_out->width];



                for (i = 0; i < pixel->nb; i++)

                    weight_sum += pixel->weights[i];



                if (!weight_sum || !pixel->nb) {

                    pixel->weights[0] = ALPHA_MAX - alpha;

                    pixel->refs[0] = 1;

                    pixel->mvs[0][0] = 0;

                    pixel->mvs[0][1] = 0;

                    pixel->weights[1] = alpha;

                    pixel->refs[1] = 2;

                    pixel->mvs[1][0] = 0;

                    pixel->mvs[1][1] = 0;

                    pixel->nb = 2;



                    weight_sum = ALPHA_MAX;

                }



                for (i = 0; i < pixel->nb; i++) {

                    Frame *frame = &mi_ctx->frames[pixel->refs[i]];

                    if (chroma) {

                        x_mv = (x >> mi_ctx->chroma_h_shift) + (pixel->mvs[i][0] >> mi_ctx->chroma_h_shift);

                        y_mv = (y >> mi_ctx->chroma_v_shift) + (pixel->mvs[i][1] >> mi_ctx->chroma_v_shift);

                    } else {

                        x_mv = x + pixel->mvs[i][0];

                        y_mv = y + pixel->mvs[i][1];

                    }



                    val += pixel->weights[i] * frame->avf->data[plane][x_mv + y_mv * frame->avf->linesize[plane]];

                }



                val = ROUNDED_DIV(val, weight_sum);



                if (chroma)

                    avf_out->data[plane][(x >> mi_ctx->chroma_h_shift) + (y >> mi_ctx->chroma_v_shift) * avf_out->linesize[plane]] = val;

                else

                    avf_out->data[plane][x + y * avf_out->linesize[plane]] = val;

            }

    }

}
