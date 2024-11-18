void tcg_target_init(TCGContext *s)

{

    tcg_regset_set32(tcg_target_available_regs[TCG_TYPE_I32], 0, 0xffffffff);

#if defined(__sparc_v9__) && !defined(__sparc_v8plus__)

    tcg_regset_set32(tcg_target_available_regs[TCG_TYPE_I64], 0, 0xffffffff);

#endif

    tcg_regset_set32(tcg_target_call_clobber_regs, 0,

                     (1 << TCG_REG_G1) |

                     (1 << TCG_REG_G2) |

                     (1 << TCG_REG_G3) |

                     (1 << TCG_REG_G4) |

                     (1 << TCG_REG_G5) |

                     (1 << TCG_REG_G6) |

                     (1 << TCG_REG_G7) |

                     (1 << TCG_REG_O0) |

                     (1 << TCG_REG_O1) |

                     (1 << TCG_REG_O2) |

                     (1 << TCG_REG_O3) |

                     (1 << TCG_REG_O4) |

                     (1 << TCG_REG_O5) |

                     (1 << TCG_REG_O7));



    tcg_regset_clear(s->reserved_regs);

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_G0);

#if defined(__sparc_v9__) && !defined(__sparc_v8plus__)

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_I4); // for internal use

#endif

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_I5); // for internal use

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_I6);

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_I7);

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_O6);

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_O7);

    tcg_add_target_add_op_defs(sparc_op_defs);

}
