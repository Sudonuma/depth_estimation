# file: process_assets.py

from comet_ml import API
import sys
api = API(api_key="l6NAe3ZOaMzGNsrPmy78yRnEv")

print("Looking up malformed figure assets in experiments...")

experiments = api.get_experiments(sys.argv[-2], sys.argv[-1])
for experiment in experiments:
    asset_list = experiment.get_asset_list()
    for asset_json in asset_list:
        if ((asset_json["type"] == "image") and
            (not asset_json["fileName"].endswith(".svg"))):
            asset = experiment.get_asset(asset_json["assetId"])
            if asset.startswith("<?xml".encode()):
                print(",".join([experiment.id, asset_json["assetId"], "\"svg\""]))
                #print(",".join([experiment.id, asset_json["assetId"], asset_"\"svg\""]))

