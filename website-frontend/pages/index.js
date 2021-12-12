import dynamic from 'next/dynamic'

import Head from 'next/head'
import { Suspense } from 'react'
import Layout from '../components/Layout'
const MainView = dynamic(() => import('../components/MainView'), { ssr: false })


export default function Home() {
  return (
    < >
      <Head>
        <title>2D to 3D Webdemo</title>
        <link rel="icon" href="/favicon.ico" />

        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap" />
      </Head>

      <Layout>
          <MainView />
      </Layout>

    </>
  )
}
