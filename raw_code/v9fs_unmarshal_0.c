size_t v9fs_unmarshal(struct iovec *out_sg, int out_num, size_t offset,

                      int bswap, const char *fmt, ...)

{

    int i;

    va_list ap;

    size_t old_offset = offset;



    va_start(ap, fmt);

    for (i = 0; fmt[i]; i++) {

        switch (fmt[i]) {

        case 'b': {

            uint8_t *valp = va_arg(ap, uint8_t *);

            offset += v9fs_unpack(valp, out_sg, out_num, offset, sizeof(*valp));

            break;

        }

        case 'w': {

            uint16_t val, *valp;

            valp = va_arg(ap, uint16_t *);

            offset += v9fs_unpack(&val, out_sg, out_num, offset, sizeof(val));

            if (bswap) {

                *valp = le16_to_cpu(val);

            } else {

                *valp = val;

            }

            break;

        }

        case 'd': {

            uint32_t val, *valp;

            valp = va_arg(ap, uint32_t *);

            offset += v9fs_unpack(&val, out_sg, out_num, offset, sizeof(val));

            if (bswap) {

                *valp = le32_to_cpu(val);

            } else {

                *valp = val;

            }

            break;

        }

        case 'q': {

            uint64_t val, *valp;

            valp = va_arg(ap, uint64_t *);

            offset += v9fs_unpack(&val, out_sg, out_num, offset, sizeof(val));

            if (bswap) {

                *valp = le64_to_cpu(val);

            } else {

                *valp = val;

            }

            break;

        }

        case 's': {

            V9fsString *str = va_arg(ap, V9fsString *);

            offset += v9fs_unmarshal(out_sg, out_num, offset, bswap,

                            "w", &str->size);

            /* FIXME: sanity check str->size */

            str->data = g_malloc(str->size + 1);

            offset += v9fs_unpack(str->data, out_sg, out_num, offset,

                            str->size);

            str->data[str->size] = 0;

            break;

        }

        case 'Q': {

            V9fsQID *qidp = va_arg(ap, V9fsQID *);

            offset += v9fs_unmarshal(out_sg, out_num, offset, bswap, "bdq",

                                     &qidp->type, &qidp->version, &qidp->path);

            break;

        }

        case 'S': {

            V9fsStat *statp = va_arg(ap, V9fsStat *);

            offset += v9fs_unmarshal(out_sg, out_num, offset, bswap,

                                    "wwdQdddqsssssddd",

                                     &statp->size, &statp->type, &statp->dev,

                                     &statp->qid, &statp->mode, &statp->atime,

                                     &statp->mtime, &statp->length,

                                     &statp->name, &statp->uid, &statp->gid,

                                     &statp->muid, &statp->extension,

                                     &statp->n_uid, &statp->n_gid,

                                     &statp->n_muid);

            break;

        }

        case 'I': {

            V9fsIattr *iattr = va_arg(ap, V9fsIattr *);

            offset += v9fs_unmarshal(out_sg, out_num, offset, bswap,

                                     "ddddqqqqq",

                                     &iattr->valid, &iattr->mode,

                                     &iattr->uid, &iattr->gid, &iattr->size,

                                     &iattr->atime_sec, &iattr->atime_nsec,

                                     &iattr->mtime_sec, &iattr->mtime_nsec);

            break;

        }

        default:

            break;

        }

    }



    va_end(ap);



    return offset - old_offset;

}
