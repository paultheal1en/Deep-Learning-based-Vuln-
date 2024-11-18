static void set_seg(struct kvm_segment *lhs, const SegmentCache *rhs)

{

    unsigned flags = rhs->flags;

    lhs->selector = rhs->selector;

    lhs->base = rhs->base;

    lhs->limit = rhs->limit;

    lhs->type = (flags >> DESC_TYPE_SHIFT) & 15;

    lhs->present = (flags & DESC_P_MASK) != 0;

    lhs->dpl = rhs->selector & 3;

    lhs->db = (flags >> DESC_B_SHIFT) & 1;

    lhs->s = (flags & DESC_S_MASK) != 0;

    lhs->l = (flags >> DESC_L_SHIFT) & 1;

    lhs->g = (flags & DESC_G_MASK) != 0;

    lhs->avl = (flags & DESC_AVL_MASK) != 0;

    lhs->unusable = 0;

}
