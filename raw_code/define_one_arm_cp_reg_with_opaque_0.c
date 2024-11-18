void define_one_arm_cp_reg_with_opaque(ARMCPU *cpu,

                                       const ARMCPRegInfo *r, void *opaque)

{

    /* Define implementations of coprocessor registers.

     * We store these in a hashtable because typically

     * there are less than 150 registers in a space which

     * is 16*16*16*8*8 = 262144 in size.

     * Wildcarding is supported for the crm, opc1 and opc2 fields.

     * If a register is defined twice then the second definition is

     * used, so this can be used to define some generic registers and

     * then override them with implementation specific variations.

     * At least one of the original and the second definition should

     * include ARM_CP_OVERRIDE in its type bits -- this is just a guard

     * against accidental use.

     */

    int crm, opc1, opc2;

    int crmmin = (r->crm == CP_ANY) ? 0 : r->crm;

    int crmmax = (r->crm == CP_ANY) ? 15 : r->crm;

    int opc1min = (r->opc1 == CP_ANY) ? 0 : r->opc1;

    int opc1max = (r->opc1 == CP_ANY) ? 7 : r->opc1;

    int opc2min = (r->opc2 == CP_ANY) ? 0 : r->opc2;

    int opc2max = (r->opc2 == CP_ANY) ? 7 : r->opc2;

    /* 64 bit registers have only CRm and Opc1 fields */

    assert(!((r->type & ARM_CP_64BIT) && (r->opc2 || r->crn)));

    /* Check that the register definition has enough info to handle

     * reads and writes if they are permitted.

     */

    if (!(r->type & (ARM_CP_SPECIAL|ARM_CP_CONST))) {

        if (r->access & PL3_R) {

            assert(r->fieldoffset || r->readfn);

        }

        if (r->access & PL3_W) {

            assert(r->fieldoffset || r->writefn);

        }

    }

    /* Bad type field probably means missing sentinel at end of reg list */

    assert(cptype_valid(r->type));

    for (crm = crmmin; crm <= crmmax; crm++) {

        for (opc1 = opc1min; opc1 <= opc1max; opc1++) {

            for (opc2 = opc2min; opc2 <= opc2max; opc2++) {

                uint32_t *key = g_new(uint32_t, 1);

                ARMCPRegInfo *r2 = g_memdup(r, sizeof(ARMCPRegInfo));

                int is64 = (r->type & ARM_CP_64BIT) ? 1 : 0;

                *key = ENCODE_CP_REG(r->cp, is64, r->crn, crm, opc1, opc2);

                if (opaque) {

                    r2->opaque = opaque;

                }

                /* Make sure reginfo passed to helpers for wildcarded regs

                 * has the correct crm/opc1/opc2 for this reg, not CP_ANY:

                 */

                r2->crm = crm;

                r2->opc1 = opc1;

                r2->opc2 = opc2;

                /* By convention, for wildcarded registers only the first

                 * entry is used for migration; the others are marked as

                 * NO_MIGRATE so we don't try to transfer the register

                 * multiple times. Special registers (ie NOP/WFI) are

                 * never migratable.

                 */

                if ((r->type & ARM_CP_SPECIAL) ||

                    ((r->crm == CP_ANY) && crm != 0) ||

                    ((r->opc1 == CP_ANY) && opc1 != 0) ||

                    ((r->opc2 == CP_ANY) && opc2 != 0)) {

                    r2->type |= ARM_CP_NO_MIGRATE;

                }



                /* Overriding of an existing definition must be explicitly

                 * requested.

                 */

                if (!(r->type & ARM_CP_OVERRIDE)) {

                    ARMCPRegInfo *oldreg;

                    oldreg = g_hash_table_lookup(cpu->cp_regs, key);

                    if (oldreg && !(oldreg->type & ARM_CP_OVERRIDE)) {

                        fprintf(stderr, "Register redefined: cp=%d %d bit "

                                "crn=%d crm=%d opc1=%d opc2=%d, "

                                "was %s, now %s\n", r2->cp, 32 + 32 * is64,

                                r2->crn, r2->crm, r2->opc1, r2->opc2,

                                oldreg->name, r2->name);

                        g_assert_not_reached();

                    }

                }

                g_hash_table_insert(cpu->cp_regs, key, r2);

            }

        }

    }

}
