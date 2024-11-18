real_parse_asm_rulebook(AVFormatContext *s, AVStream *orig_st,

                        const char *p)

{

    const char *end;

    int n_rules, odd = 0;

    AVStream *st;



    /**

     * The ASMRuleBook contains a list of comma-separated strings per rule,

     * and each rule is separated by a ;. The last one also has a ; at the

     * end so we can use it as delimiter.

     * Every rule occurs twice, once for when the RTSP packet header marker

     * is set and once for if it isn't. We only read the first because we

     * don't care much (that's what the "odd" variable is for).

     * Each rule contains a set of one or more statements, optionally

     * preceeded by a single condition. If there's a condition, the rule

     * starts with a '#'. Multiple conditions are merged between brackets,

     * so there are never multiple conditions spread out over separate

     * statements. Generally, these conditions are bitrate limits (min/max)

     * for multi-bitrate streams.

     */

    if (*p == '\"') p++;

    for (n_rules = 0; s->nb_streams < MAX_STREAMS;) {

        if (!(end = strchr(p, ';')))

            break;

        if (!odd && end != p) {

            if (n_rules > 0)

                st = add_dstream(s, orig_st);

            else

                st = orig_st;

            real_parse_asm_rule(st, p, end);

            n_rules++;

        }

        p = end + 1;

        odd ^= 1;

    }

}
