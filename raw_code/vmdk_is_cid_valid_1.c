static int vmdk_is_cid_valid(BlockDriverState *bs)

{

    BDRVVmdkState *s = bs->opaque;

    uint32_t cur_pcid;



    if (!s->cid_checked && bs->backing) {

        BlockDriverState *p_bs = bs->backing->bs;



        cur_pcid = vmdk_read_cid(p_bs, 0);

        if (s->parent_cid != cur_pcid) {

            /* CID not valid */

            return 0;

        }

    }

    s->cid_checked = true;

    /* CID valid */

    return 1;

}
