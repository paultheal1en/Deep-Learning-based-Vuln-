int qcow2_encrypt_sectors(BDRVQcow2State *s, int64_t sector_num,

                          uint8_t *buf, int nb_sectors, bool enc,

                          Error **errp)

{

    union {

        uint64_t ll[2];

        uint8_t b[16];

    } ivec;

    int i;

    int ret;



    for(i = 0; i < nb_sectors; i++) {

        ivec.ll[0] = cpu_to_le64(sector_num);

        ivec.ll[1] = 0;

        if (qcrypto_cipher_setiv(s->cipher,

                                 ivec.b, G_N_ELEMENTS(ivec.b),

                                 errp) < 0) {

            return -1;

        }

        if (enc) {

            ret = qcrypto_cipher_encrypt(s->cipher,

                                         buf, buf,

                                         512,

                                         errp);

        } else {

            ret = qcrypto_cipher_decrypt(s->cipher,

                                         buf, buf,

                                         512,

                                         errp);

        }

        if (ret < 0) {

            return -1;

        }

        sector_num++;

        buf += 512;

    }

    return 0;

}
