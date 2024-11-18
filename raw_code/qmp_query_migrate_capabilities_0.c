MigrationCapabilityStatusList *qmp_query_migrate_capabilities(Error **errp)

{

    MigrationCapabilityStatusList *head = NULL;

    MigrationCapabilityStatusList *caps;

    MigrationState *s = migrate_get_current();

    int i;



    caps = NULL; /* silence compiler warning */

    for (i = 0; i < MIGRATION_CAPABILITY__MAX; i++) {

#ifndef CONFIG_LIVE_BLOCK_MIGRATION

        if (i == MIGRATION_CAPABILITY_BLOCK) {

            continue;

        }

#endif

        if (i == MIGRATION_CAPABILITY_X_COLO && !colo_supported()) {

            continue;

        }

        if (head == NULL) {

            head = g_malloc0(sizeof(*caps));

            caps = head;

        } else {

            caps->next = g_malloc0(sizeof(*caps));

            caps = caps->next;

        }

        caps->value =

            g_malloc(sizeof(*caps->value));

        caps->value->capability = i;

        caps->value->state = s->enabled_capabilities[i];

    }



    return head;

}
