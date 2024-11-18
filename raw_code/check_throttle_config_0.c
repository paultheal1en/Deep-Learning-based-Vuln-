static bool check_throttle_config(ThrottleConfig *cfg, Error **errp)

{

    if (throttle_conflicting(cfg, errp)) {

        return false;

    }



    if (!throttle_is_valid(cfg, errp)) {

        return false;

    }



    if (throttle_max_is_missing_limit(cfg, errp)) {

        return false;

    }



    return true;

}
