static int QEMU_WARN_UNUSED_RESULT update_refcount(BlockDriverState *bs,

    int64_t offset, int64_t length, int addend)

{

    BDRVQcowState *s = bs->opaque;

    int64_t start, last, cluster_offset;

    int64_t refcount_block_offset = 0;

    int64_t table_index = -1, old_table_index;

    int first_index = -1, last_index = -1;

    int ret;



#ifdef DEBUG_ALLOC2

    printf("update_refcount: offset=%" PRId64 " size=%" PRId64 " addend=%d\n",

           offset, length, addend);

#endif

    if (length < 0) {

        return -EINVAL;

    } else if (length == 0) {

        return 0;

    }



    start = offset & ~(s->cluster_size - 1);

    last = (offset + length - 1) & ~(s->cluster_size - 1);

    for(cluster_offset = start; cluster_offset <= last;

        cluster_offset += s->cluster_size)

    {

        int block_index, refcount;

        int64_t cluster_index = cluster_offset >> s->cluster_bits;

        int64_t new_block;



        /* Only write refcount block to disk when we are done with it */

        old_table_index = table_index;

        table_index = cluster_index >> (s->cluster_bits - REFCOUNT_SHIFT);

        if ((old_table_index >= 0) && (table_index != old_table_index)) {



            ret = write_refcount_block_entries(bs, refcount_block_offset,

                first_index, last_index);

            if (ret < 0) {

                return ret;

            }



            first_index = -1;

            last_index = -1;

        }



        /* Load the refcount block and allocate it if needed */

        new_block = alloc_refcount_block(bs, cluster_index);

        if (new_block < 0) {

            ret = new_block;

            goto fail;

        }

        refcount_block_offset = new_block;



        /* we can update the count and save it */

        block_index = cluster_index &

            ((1 << (s->cluster_bits - REFCOUNT_SHIFT)) - 1);

        if (first_index == -1 || block_index < first_index) {

            first_index = block_index;

        }

        if (block_index > last_index) {

            last_index = block_index;

        }



        refcount = be16_to_cpu(s->refcount_block_cache[block_index]);

        refcount += addend;

        if (refcount < 0 || refcount > 0xffff) {

            ret = -EINVAL;

            goto fail;

        }

        if (refcount == 0 && cluster_index < s->free_cluster_index) {

            s->free_cluster_index = cluster_index;

        }

        s->refcount_block_cache[block_index] = cpu_to_be16(refcount);

    }



    ret = 0;

fail:



    /* Write last changed block to disk */

    if (refcount_block_offset != 0) {

        int wret;

        wret = write_refcount_block_entries(bs, refcount_block_offset,

            first_index, last_index);

        if (wret < 0) {

            return ret < 0 ? ret : wret;

        }

    }



    /*

     * Try do undo any updates if an error is returned (This may succeed in

     * some cases like ENOSPC for allocating a new refcount block)

     */

    if (ret < 0) {

        int dummy;

        dummy = update_refcount(bs, offset, cluster_offset - offset, -addend);

    }



    bdrv_flush(bs->file);



    return ret;

}
