static void decode_v1_vector(CinepakEncContext *s, AVPicture *sub_pict, mb_info *mb, strip_info *info)

{

    int entry_size = s->pix_fmt == AV_PIX_FMT_YUV420P ? 6 : 4;



    sub_pict->data[0][0] =

            sub_pict->data[0][1] =

            sub_pict->data[0][    sub_pict->linesize[0]] =

            sub_pict->data[0][1+  sub_pict->linesize[0]] = info->v1_codebook[mb->v1_vector*entry_size];



    sub_pict->data[0][2] =

            sub_pict->data[0][3] =

            sub_pict->data[0][2+  sub_pict->linesize[0]] =

            sub_pict->data[0][3+  sub_pict->linesize[0]] = info->v1_codebook[mb->v1_vector*entry_size+1];



    sub_pict->data[0][2*sub_pict->linesize[0]] =

            sub_pict->data[0][1+2*sub_pict->linesize[0]] =

            sub_pict->data[0][  3*sub_pict->linesize[0]] =

            sub_pict->data[0][1+3*sub_pict->linesize[0]] = info->v1_codebook[mb->v1_vector*entry_size+2];



    sub_pict->data[0][2+2*sub_pict->linesize[0]] =

            sub_pict->data[0][3+2*sub_pict->linesize[0]] =

            sub_pict->data[0][2+3*sub_pict->linesize[0]] =

            sub_pict->data[0][3+3*sub_pict->linesize[0]] = info->v1_codebook[mb->v1_vector*entry_size+3];



    if(s->pix_fmt == AV_PIX_FMT_YUV420P) {

        sub_pict->data[1][0] =

            sub_pict->data[1][1] =

            sub_pict->data[1][    sub_pict->linesize[1]] =

            sub_pict->data[1][1+  sub_pict->linesize[1]] = info->v1_codebook[mb->v1_vector*entry_size+4];



        sub_pict->data[2][0] =

            sub_pict->data[2][1] =

            sub_pict->data[2][    sub_pict->linesize[2]] =

            sub_pict->data[2][1+  sub_pict->linesize[2]] = info->v1_codebook[mb->v1_vector*entry_size+5];

    }

}
