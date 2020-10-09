module.exports = {
    apps: [
        {
            name: 'aws-codedeploy',
            script: 'streamlit',
            args: 'run explore.py',
            interpreter: 'none',
            env: {
                NODE_ENV: 'development',
            },
        },
    ],
}