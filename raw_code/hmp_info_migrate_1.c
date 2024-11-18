void hmp_info_migrate(Monitor *mon, const QDict *qdict)

{

    MigrationInfo *info;

    MigrationCapabilityStatusList *caps, *cap;



    info = qmp_query_migrate(NULL);

    caps = qmp_query_migrate_capabilities(NULL);



    /* do not display parameters during setup */

    if (info->has_status && caps) {

        monitor_printf(mon, "capabilities: ");

        for (cap = caps; cap; cap = cap->next) {

            monitor_printf(mon, "%s: %s ",

                           MigrationCapability_lookup[cap->value->capability],

                           cap->value->state ? "on" : "off");

        }

        monitor_printf(mon, "\n");

    }



    if (info->has_status) {

        monitor_printf(mon, "Migration status: %s\n",

                       MigrationStatus_lookup[info->status]);

        monitor_printf(mon, "total time: %" PRIu64 " milliseconds\n",

                       info->total_time);

        if (info->has_expected_downtime) {

            monitor_printf(mon, "expected downtime: %" PRIu64 " milliseconds\n",

                           info->expected_downtime);

        }

        if (info->has_downtime) {

            monitor_printf(mon, "downtime: %" PRIu64 " milliseconds\n",

                           info->downtime);

        }

        if (info->has_setup_time) {

            monitor_printf(mon, "setup: %" PRIu64 " milliseconds\n",

                           info->setup_time);

        }

    }



    if (info->has_ram) {

        monitor_printf(mon, "transferred ram: %" PRIu64 " kbytes\n",

                       info->ram->transferred >> 10);

        monitor_printf(mon, "throughput: %0.2f mbps\n",

                       info->ram->mbps);

        monitor_printf(mon, "remaining ram: %" PRIu64 " kbytes\n",

                       info->ram->remaining >> 10);

        monitor_printf(mon, "total ram: %" PRIu64 " kbytes\n",

                       info->ram->total >> 10);

        monitor_printf(mon, "duplicate: %" PRIu64 " pages\n",

                       info->ram->duplicate);

        monitor_printf(mon, "skipped: %" PRIu64 " pages\n",

                       info->ram->skipped);

        monitor_printf(mon, "normal: %" PRIu64 " pages\n",

                       info->ram->normal);

        monitor_printf(mon, "normal bytes: %" PRIu64 " kbytes\n",

                       info->ram->normal_bytes >> 10);

        monitor_printf(mon, "dirty sync count: %" PRIu64 "\n",

                       info->ram->dirty_sync_count);

        if (info->ram->dirty_pages_rate) {

            monitor_printf(mon, "dirty pages rate: %" PRIu64 " pages\n",

                           info->ram->dirty_pages_rate);

        }

    }



    if (info->has_disk) {

        monitor_printf(mon, "transferred disk: %" PRIu64 " kbytes\n",

                       info->disk->transferred >> 10);

        monitor_printf(mon, "remaining disk: %" PRIu64 " kbytes\n",

                       info->disk->remaining >> 10);

        monitor_printf(mon, "total disk: %" PRIu64 " kbytes\n",

                       info->disk->total >> 10);

    }



    if (info->has_xbzrle_cache) {

        monitor_printf(mon, "cache size: %" PRIu64 " bytes\n",

                       info->xbzrle_cache->cache_size);

        monitor_printf(mon, "xbzrle transferred: %" PRIu64 " kbytes\n",

                       info->xbzrle_cache->bytes >> 10);

        monitor_printf(mon, "xbzrle pages: %" PRIu64 " pages\n",

                       info->xbzrle_cache->pages);

        monitor_printf(mon, "xbzrle cache miss: %" PRIu64 "\n",

                       info->xbzrle_cache->cache_miss);

        monitor_printf(mon, "xbzrle cache miss rate: %0.2f\n",

                       info->xbzrle_cache->cache_miss_rate);

        monitor_printf(mon, "xbzrle overflow : %" PRIu64 "\n",

                       info->xbzrle_cache->overflow);

    }



    if (info->has_cpu_throttle_percentage) {

        monitor_printf(mon, "cpu throttle percentage: %" PRIu64 "\n",

                       info->cpu_throttle_percentage);

    }



    qapi_free_MigrationInfo(info);

    qapi_free_MigrationCapabilityStatusList(caps);

}
