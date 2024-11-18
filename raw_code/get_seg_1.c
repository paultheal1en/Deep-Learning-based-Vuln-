static void get_seg(SegmentCache *lhs, const struct kvm_segment *rhs)

{

    lhs->selector = rhs->selector;

    lhs->base = rhs->base;

    lhs->limit = rhs->limit;

    lhs->flags = (rhs->type << DESC_TYPE_SHIFT) |

                 (rhs->present * DESC_P_MASK) |

                 (rhs->dpl << DESC_DPL_SHIFT) |

                 (rhs->db << DESC_B_SHIFT) |

                 (rhs->s * DESC_S_MASK) |

                 (rhs->l << DESC_L_SHIFT) |

                 (rhs->g * DESC_G_MASK) |

                 (rhs->avl * DESC_AVL_MASK);

}
