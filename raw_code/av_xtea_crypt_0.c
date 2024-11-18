void av_xtea_crypt(AVXTEA *ctx, uint8_t *dst, const uint8_t *src, int count,

                   uint8_t *iv, int decrypt)

{

    int i;



    while (count > 0) {

        if (decrypt) {

            xtea_crypt_ecb(ctx, dst, src, decrypt);



            if (iv) {

                for (i = 0; i < 8; i++)

                    dst[i] = dst[i] ^ iv[i];

                memcpy(iv, src, 8);

            }

        } else {

            if (iv) {

                for (i = 0; i < 8; i++)

                    dst[i] = src[i] ^ iv[i];

                xtea_crypt_ecb(ctx, dst, dst, decrypt);

                memcpy(iv, dst, 8);

            } else {

                xtea_crypt_ecb(ctx, dst, src, decrypt);

            }

        }



        src   += 8;

        dst   += 8;

        count -= 8;

    }

}
