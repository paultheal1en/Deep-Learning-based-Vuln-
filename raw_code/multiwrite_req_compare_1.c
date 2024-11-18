static int multiwrite_req_compare(const void *a, const void *b)

{

    return (((BlockRequest*) a)->sector - ((BlockRequest*) b)->sector);

}
