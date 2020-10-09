module.exports = {
    apps: [
        {
            name: 'aws-codedeploy',
            script: 'npx',
            args: 'serve -s build -l 3000 -n',
            interpreter: 'none',
            env: {
                NODE_ENV: 'development',
            },
        },
    ],
}