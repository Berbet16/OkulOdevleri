from flask import Flask
from ec2_metadata import ec2_metadata
from flask import request

app = Flask(__name__)


@app.route('/')
def hello_world():
    instanceID = ec2_metadata.instance_id
    amiLaunchIndex = ec2_metadata.ami_launch_index
    publicHostname = ec2_metadata.public_hostname
    publicIpv4 = ec2_metadata.public_ipv4
    localHostname = ec2_metadata.private_hostname
    localIpv4 = ec2_metadata.private_ipv4

    resultString = "<table><thead><tr><th>Metadata</th><th>Value</th></tr></thead><tbody><tr><td>instance id</td><td>" + instanceID + "</td></tr><tr><td>ami launch index</td><td>" + str(
        amiLaunchIndex) + "</td></tr><tr><td>public hostname</td><td>" + publicHostname + "</td></tr><tr><td>public ipv4</td><td>" + publicIpv4 + "</td></tr><tr><td>local hostname</td><td>" + localHostname + "</td></tr><tr><td>local ipv4</td><td>" + localIpv4 + "</td></tr></tbody></table>"

    return resultString


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)