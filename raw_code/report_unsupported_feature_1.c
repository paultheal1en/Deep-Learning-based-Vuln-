static void report_unsupported_feature(BlockDriverState *bs,

    Error **errp, Qcow2Feature *table, uint64_t mask)

{

    while (table && table->name[0] != '\0') {

        if (table->type == QCOW2_FEAT_TYPE_INCOMPATIBLE) {

            if (mask & (1 << table->bit)) {

                report_unsupported(bs, errp, "%.46s", table->name);

                mask &= ~(1 << table->bit);

            }

        }

        table++;

    }



    if (mask) {

        report_unsupported(bs, errp, "Unknown incompatible feature: %" PRIx64,

                           mask);

    }

}
