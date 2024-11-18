static int vmdk_is_cid_valid(BlockDriverState *bs)

{

#ifdef CHECK_CID

    BDRVVmdkState *s = bs->opaque;

    BlockDriverState *p_bs = bs->backing_hd;

    uint32_t cur_pcid;



    if (p_bs) {

        cur_pcid = vmdk_read_cid(p_bs, 0);

        if (s->parent_cid != cur_pcid) {

            /* CID not valid */

            return 0;

        }

    }

#endif

    /* CID valid */

    return 1;

}
