size_t v9fs_marshal(struct iovec *in_sg, int in_num, size_t offset,

                    int bswap, const char *fmt, ...)

{

    int i;

    va_list ap;

    size_t old_offset = offset;



    va_start(ap, fmt);

    for (i = 0; fmt[i]; i++) {

        switch (fmt[i]) {

        case 'b': {

            uint8_t val = va_arg(ap, int);

            offset += v9fs_pack(in_sg, in_num, offset, &val, sizeof(val));

            break;

        }

        case 'w': {

            uint16_t val;

            if (bswap) {

                cpu_to_le16w(&val, va_arg(ap, int));

            } else {

                val =  va_arg(ap, int);

            }

            offset += v9fs_pack(in_sg, in_num, offset, &val, sizeof(val));

            break;

        }

        case 'd': {

            uint32_t val;

            if (bswap) {

                cpu_to_le32w(&val, va_arg(ap, uint32_t));

            } else {

                val =  va_arg(ap, uint32_t);

            }

            offset += v9fs_pack(in_sg, in_num, offset, &val, sizeof(val));

            break;

        }

        case 'q': {

            uint64_t val;

            if (bswap) {

                cpu_to_le64w(&val, va_arg(ap, uint64_t));

            } else {

                val =  va_arg(ap, uint64_t);

            }

            offset += v9fs_pack(in_sg, in_num, offset, &val, sizeof(val));

            break;

        }

        case 's': {

            V9fsString *str = va_arg(ap, V9fsString *);

            offset += v9fs_marshal(in_sg, in_num, offset, bswap,

                            "w", str->size);

            offset += v9fs_pack(in_sg, in_num, offset, str->data, str->size);

            break;

        }

        case 'Q': {

            V9fsQID *qidp = va_arg(ap, V9fsQID *);

            offset += v9fs_marshal(in_sg, in_num, offset, bswap, "bdq",

                                   qidp->type, qidp->version, qidp->path);

            break;

        }

        case 'S': {

            V9fsStat *statp = va_arg(ap, V9fsStat *);

            offset += v9fs_marshal(in_sg, in_num, offset, bswap,

                                   "wwdQdddqsssssddd",

                                   statp->size, statp->type, statp->dev,

                                   &statp->qid, statp->mode, statp->atime,

                                   statp->mtime, statp->length, &statp->name,

                                   &statp->uid, &statp->gid, &statp->muid,

                                   &statp->extension, statp->n_uid,

                                   statp->n_gid, statp->n_muid);

            break;

        }

        case 'A': {

            V9fsStatDotl *statp = va_arg(ap, V9fsStatDotl *);

            offset += v9fs_marshal(in_sg, in_num, offset, bswap,

                                   "qQdddqqqqqqqqqqqqqqq",

                                   statp->st_result_mask,

                                   &statp->qid, statp->st_mode,

                                   statp->st_uid, statp->st_gid,

                                   statp->st_nlink, statp->st_rdev,

                                   statp->st_size, statp->st_blksize,

                                   statp->st_blocks, statp->st_atime_sec,

                                   statp->st_atime_nsec, statp->st_mtime_sec,

                                   statp->st_mtime_nsec, statp->st_ctime_sec,

                                   statp->st_ctime_nsec, statp->st_btime_sec,

                                   statp->st_btime_nsec, statp->st_gen,

                                   statp->st_data_version);

            break;

        }

        default:

            break;

        }

    }

    va_end(ap);



    return offset - old_offset;

}
