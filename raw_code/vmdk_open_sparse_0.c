static int vmdk_open_sparse(BlockDriverState *bs,

                            BlockDriverState *file, int flags,

                            char *buf, Error **errp)

{

    uint32_t magic;



    magic = ldl_be_p(buf);

    switch (magic) {

        case VMDK3_MAGIC:

            return vmdk_open_vmfs_sparse(bs, file, flags, errp);

            break;

        case VMDK4_MAGIC:

            return vmdk_open_vmdk4(bs, file, flags, errp);

            break;

        default:

            return -EMEDIUMTYPE;

            break;

    }

}
