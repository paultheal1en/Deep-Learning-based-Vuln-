static int raw_pwrite(BlockDriverState *bs, int64_t offset,

                      const uint8_t *buf, int count)

{

    BDRVRawState *s = bs->opaque;

    int size, ret, shift, sum;



    sum = 0;



    if (s->aligned_buf != NULL) {



        if (offset & 0x1ff) {

            /* align offset on a 512 bytes boundary */

            shift = offset & 0x1ff;

            ret = raw_pread_aligned(bs, offset - shift, s->aligned_buf, 512);

            if (ret < 0)

                return ret;



            size = 512 - shift;

            if (size > count)

                size = count;

            memcpy(s->aligned_buf + shift, buf, size);



            ret = raw_pwrite_aligned(bs, offset - shift, s->aligned_buf, 512);

            if (ret < 0)

                return ret;



            buf += size;

            offset += size;

            count -= size;

            sum += size;



            if (count == 0)

                return sum;

        }

        if (count & 0x1ff || (uintptr_t) buf & 0x1ff) {



            while ((size = (count & ~0x1ff)) != 0) {



                if (size > ALIGNED_BUFFER_SIZE)

                    size = ALIGNED_BUFFER_SIZE;



                memcpy(s->aligned_buf, buf, size);



                ret = raw_pwrite_aligned(bs, offset, s->aligned_buf, size);

                if (ret < 0)

                    return ret;



                buf += ret;

                offset += ret;

                count -= ret;

                sum += ret;

            }

            /* here, count < 512 because (count & ~0x1ff) == 0 */

            if (count) {

                ret = raw_pread_aligned(bs, offset, s->aligned_buf, 512);

                if (ret < 0)

                    return ret;

                 memcpy(s->aligned_buf, buf, count);



                 ret = raw_pwrite_aligned(bs, offset, s->aligned_buf, 512);

                 if (ret < 0)

                     return ret;

                 if (count < ret)

                     ret = count;



                 sum += ret;

            }

            return sum;

        }

    }

    return raw_pwrite_aligned(bs, offset, buf, count) + sum;

}
