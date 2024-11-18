static void decode_v4_vector(CinepakEncContext *s, AVPicture *sub_pict, int *v4_vector, strip_info *info)

{

    int i, x, y, entry_size = s->pix_fmt == AV_PIX_FMT_YUV420P ? 6 : 4;



    for(i = y = 0; y < 4; y += 2) {

        for(x = 0; x < 4; x += 2, i++) {

            sub_pict->data[0][x   +     y*sub_pict->linesize[0]] = info->v4_codebook[v4_vector[i]*entry_size];

            sub_pict->data[0][x+1 +     y*sub_pict->linesize[0]] = info->v4_codebook[v4_vector[i]*entry_size+1];

            sub_pict->data[0][x   + (y+1)*sub_pict->linesize[0]] = info->v4_codebook[v4_vector[i]*entry_size+2];

            sub_pict->data[0][x+1 + (y+1)*sub_pict->linesize[0]] = info->v4_codebook[v4_vector[i]*entry_size+3];



            if(s->pix_fmt == AV_PIX_FMT_YUV420P) {

                sub_pict->data[1][(x>>1) + (y>>1)*sub_pict->linesize[1]] = info->v4_codebook[v4_vector[i]*entry_size+4];

                sub_pict->data[2][(x>>1) + (y>>1)*sub_pict->linesize[2]] = info->v4_codebook[v4_vector[i]*entry_size+5];

            }

        }

    }

}
