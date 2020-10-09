module.exports = {
    apps: [
        {
            name: 'aml-streamlit',
            script: 'explore.py',
            args: 'run',
            interpreter: 'streamlit',
            env: {
                NODE_ENV: 'development',
            },
        },
    ],
}